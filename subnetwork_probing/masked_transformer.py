import logging
import math
from typing import Callable, ContextManager, List, Tuple, TypeAlias

import torch
from einops import rearrange
from jaxtyping import Num
from torch.utils.checkpoint import checkpoint
from transformer_lens import ActivationCache, HookedTransformer
from transformer_lens.hook_points import HookPoint

from acdc.TLACDCCorrespondence import TLACDCCorrespondence
from acdc.TLACDCEdge import HookPointName, TorchIndex

logger = logging.getLogger(__name__)

PatchData: TypeAlias = Num[torch.Tensor, "batch pos"] | None  # use None if you want zero ablation


def create_mask_parameters_and_forward_cache_hook_points(
    use_pos_embed: bool, num_heads: int, n_layers: int, device: str, mask_init_constant: float, attn_only: bool
):
    """
    Given the relevant configuration for a transformer, this function produces two things:

    1. The parameters for the masks, in a dict that maps a hook point name to its mask parameter
    2. The hook points that need to be cached on a forward pass.

    TODO: why don't we keep a friggin dict or something to keep track of how each IndexedHookPoint maps to an index?
    basically dict[IndexedHookPoint, int] or something like that
    """

    ordered_forward_cache_hook_points: list[HookPointName] = []

    # we need to track the number of outputs so far, because this will be the number of parents
    # of all the following units.
    num_output_units_so_far = 0

    hook_point_to_parents: dict[HookPointName, list[HookPointName]] = {}
    mask_parameter_list = torch.nn.ParameterList()
    mask_parameter_dict: dict[HookPointName, torch.nn.Parameter] = {}

    # Implementation details:
    #
    # We distinguish hook points that are used for input, and hook points that are used for output.
    #
    # The input hook points are the ones that get mask parameters.
    # The output hook points are the ones that we cache on a forward pass, so that we can later mask them.

    def setup_output_hook_point(mask_name: str, num_instances: int):
        ordered_forward_cache_hook_points.append(mask_name)

        nonlocal num_output_units_so_far
        num_output_units_so_far += num_instances

    def setup_input_hook_point(mask_name: str, num_instances: int):
        """
        Adds a mask logit for the given mask name and parent nodes
        Parent nodes are (attention, MLP)

        We need to add a parameter to mask the input to these units
        """
        nonlocal num_output_units_so_far
        hook_point_to_parents[mask_name] = ordered_forward_cache_hook_points[:]  # everything that has come before

        new_mask_parameter = torch.nn.Parameter(
            torch.full((num_output_units_so_far, num_instances), mask_init_constant, device=device)
        )
        mask_parameter_list.append(new_mask_parameter)
        mask_parameter_dict[mask_name] = new_mask_parameter

    for embedding_hook_point in ["hook_embed", "hook_pos_embed"] if use_pos_embed else ["blocks.0.hook_resid_pre"]:
        setup_output_hook_point(embedding_hook_point, 1)

    # Add mask logits for ablation cache
    # Mask logits have a variable dimension depending on the number of in-edges (increases with layer)
    for layer_i in range(n_layers):
        for q_k_v in ["q", "k", "v"]:
            setup_input_hook_point(mask_name=f"blocks.{layer_i}.hook_{q_k_v}_input", num_instances=num_heads)

        setup_output_hook_point(f"blocks.{layer_i}.attn.hook_result", num_heads)

        if not attn_only:
            setup_input_hook_point(mask_name=f"blocks.{layer_i}.hook_mlp_in", num_instances=1)
            setup_output_hook_point(f"blocks.{layer_i}.hook_mlp_out", num_instances=1)

    # why does this get a mask? isn't it pointless to mask this?
    setup_input_hook_point(mask_name=f"blocks.{n_layers - 1}.hook_resid_post", num_instances=1)

    # TODO: remove this unnecessary test; move this to test
    for parent_list in hook_point_to_parents.values():
        for parent in parent_list:
            assert parent in ordered_forward_cache_hook_points, f"{parent} not in forward cache names"

    return ordered_forward_cache_hook_points, hook_point_to_parents, mask_parameter_list, mask_parameter_dict


class EdgeLevelMaskedTransformer(torch.nn.Module):
    """
    A wrapper around HookedTransformer that allows edge-level subnetwork probing.

    There are two sets of hooks:
    - `activation_mask_hook`s change the input to a node. The input to a node is the sum
      of several residual stream terms; ablated edges are looked up from `ablation_cache`
      and non-ablated edges from `forward_cache`, then the sum is taken.
    - `caching_hook`s save the output of a node to `forward_cache` for use in later layers.

    Qs:

    - what are the names of the mask parameters in the mask parameter dict? We just use the names of the hook point
    - how are all the mask params laid out as a tensor?
    - does everything use the HookName as a kind ...? if so, use that in types
    """

    model: HookedTransformer
    ablation_cache: ActivationCache
    forward_cache: ActivationCache
    hook_point_to_parents: dict[HookPointName, list[HookPointName]]  # the parents of each hook point
    mask_parameter_list: torch.nn.ParameterList  # the parameters that we use to mask the input to each node
    _mask_parameter_dict: dict[
        str, torch.nn.Parameter
    ]  # same parameters, but indexed by the hook point that they are applied to
    forward_cache_hook_points: list[
        HookPointName
    ]  # the hook points where we need to cache the output on a forward pass

    def __init__(
        self,
        model: HookedTransformer,
        beta=2 / 3,
        gamma=-0.1,
        zeta=1.1,
        mask_init_p=0.9,
        use_pos_embed=False,
        no_ablate=False,
        verbose=False,
    ):
        super().__init__()

        self.model = model
        self.n_heads = model.cfg.n_heads
        self.n_mlp = 0 if model.cfg.attn_only else 1
        self.no_ablate = no_ablate
        if no_ablate:
            print("WARNING: no_ablate is True, this is for testing only")
        self.device = self.model.parameters().__next__().device
        self.use_pos_embed = use_pos_embed
        self.verbose = verbose

        self.ablation_cache = ActivationCache({}, self.model)
        self.forward_cache = ActivationCache({}, self.model)
        # Hyperparameters
        self.beta = beta
        self.gamma = gamma
        self.zeta = zeta
        self.mask_init_p = mask_init_p

        # Copied from subnetwork probing code. Similar to log odds (but not the same)
        p = (self.mask_init_p - self.gamma) / (self.zeta - self.gamma)

        model.cfg.use_hook_mlp_in = True  # We need to hook the MLP input to do subnetwork probing

        (
            self.forward_cache_hook_points,
            self.hook_point_to_parents,
            self.mask_parameter_list,
            self._mask_parameter_dict,
        ) = create_mask_parameters_and_forward_cache_hook_points(
            use_pos_embed=self.use_pos_embed,
            num_heads=self.n_heads,
            n_layers=model.cfg.n_layers,
            device=self.device,
            mask_init_constant=math.log(p / (1 - p)),
            attn_only=model.cfg.attn_only,
        )

    @property
    def mask_logits_names(self):
        return self._mask_parameter_dict.keys()

    def sample_mask(self, mask_name: str) -> torch.Tensor:
        """Samples a binary-ish mask from the mask_scores for the particular `mask_name` activation"""
        mask_parameters = self._mask_parameter_dict[mask_name]
        uniform_sample = torch.zeros_like(mask_parameters, requires_grad=False).uniform_().clamp_(0.0001, 0.9999)
        s = torch.sigmoid((uniform_sample.log() - (1 - uniform_sample).log() + mask_parameters) / self.beta)
        s_bar = s * (self.zeta - self.gamma) + self.gamma
        mask = s_bar.clamp(min=0.0, max=1.0)

        return mask

    def regularization_loss(self) -> torch.Tensor:
        center = self.beta * math.log(-self.gamma / self.zeta)
        per_parameter_loss = [torch.sigmoid(scores - center).mean() for scores in self.mask_parameter_list]
        return torch.mean(torch.stack(per_parameter_loss))

    def _calculate_and_store_zero_ablation_cache(self) -> None:
        """Caches zero for every possible mask point."""
        patch_data = torch.zeros((1, 1), device=self.device, dtype=torch.int64)  # batch pos
        self._calculate_and_store_resampling_ablation_cache(
            patch_data
        )  # wtf? is this just to initialize the cache object? if we had tests, I would refactor this
        self.ablation_cache.cache_dict = {
            name: torch.zeros_like(scores) for name, scores in self.ablation_cache.cache_dict.items()
        }

    def _calculate_and_store_resampling_ablation_cache(self, patch_data: Num[torch.Tensor, "batch pos"]) -> None:
        # Only cache the tensors needed to fill the masked out positions
        with torch.no_grad():
            model_out, self.ablation_cache = self.model.run_with_cache(
                patch_data,
                names_filter=lambda name: name in self.forward_cache_hook_points,
                return_cache_object=True,
            )

    def calculate_and_store_ablation_cache(self, patch_data: PatchData):
        """Use None for the patch data for zero ablation."""
        if patch_data is None:
            self._calculate_and_store_zero_ablation_cache()
        else:
            assert isinstance(patch_data, torch.Tensor)
            self._calculate_and_store_resampling_ablation_cache(patch_data)

    def get_activation_values(
        self, parent_names: list[str], cache: ActivationCache
    ) -> Num[torch.Tensor, "batch seq parentindex d"]:
        """
        Returns a single tensor of the mask values used for a given hook.
        Attention is shape batch, seq, heads, head_size while MLP out is batch, seq, d_model
        so we need to reshape things to match.

        The output is "batch seq parentindex d_model", where "parentindex" is the index in the list
        of `parent_names`.
        """
        result = []
        for name in parent_names:
            value = cache[name]  # b s n_heads d, or b s d
            if value.ndim == 3:
                value = value.unsqueeze(2)  # b s 1 d
            result.append(value)
        return torch.cat(result, dim=2)

    def compute_weighted_values(self, hook: HookPoint):
        parent_names = self.hook_point_to_parents[hook.name]
        ablation_values = self.get_activation_values(parent_names, self.ablation_cache)  # b s i d
        forward_values = self.get_activation_values(parent_names, self.forward_cache)  # b s i d
        mask = self.sample_mask(hook.name)  # in_edges, nodes_per_mask, ...

        weighted_ablation_values = torch.einsum("b s i d, i o -> b s o d", ablation_values, 1 - mask)
        weighted_forward_values = torch.einsum("b s i d, i o -> b s o d", forward_values, mask)
        return weighted_ablation_values + weighted_forward_values

    def activation_mask_hook(self, hook_point_out: torch.Tensor, hook: HookPoint, verbose=False):
        """
        For edge-level SP, we discard the hook_point_out value and resum the residual stream.
        """
        show = print if verbose else lambda *args, **kwargs: None
        show(f"Doing ablation of {hook.name}")
        mem1 = torch.cuda.memory_allocated()
        show(f"Using memory {mem1:_} bytes at hook start")
        is_attn = "mlp" not in hook.name and "resid_post" not in hook.name

        # memory optimization
        out = checkpoint(self.compute_weighted_values, hook, use_reentrant=False)
        if not is_attn:
            out = rearrange(out, "b s 1 d -> b s d")

        # add back attention bias
        # Explanation: the attention bias is not part of the cached values (why not?), so we need to add it back here
        # we need to iterate over all attention layers that come before current layer
        current_block_index = int(hook.name.split(".")[1])
        last_attention_block_index = (
            current_block_index + 1 if ("resid_post" in hook.name or "mlp" in hook.name) else current_block_index
        )
        for layer in self.model.blocks[:last_attention_block_index]:
            out += layer.attn.b_O

        if self.no_ablate and not torch.allclose(hook_point_out, out, atol=1e-4):
            print(f"Warning: hook_point_out and out are not close for {hook.name}")
            print(f"{hook_point_out.mean()=}, {out.mean()=}")

        if self.verbose:
            no_change = torch.allclose(hook_point_out, out)
            absdiff = (hook_point_out - out).abs().mean()
            # sqdiff_values = (a_values - f_values).pow(2).mean()
            print(f"Ablation hook {'did NOT' if no_change else 'DID'} change {hook.name} by {absdiff:.3f}")
        torch.cuda.empty_cache()
        show(f"Using memory {torch.cuda.memory_allocated():_} bytes after clearing cache")
        return out

    def caching_hook(self, hook_point_out: torch.Tensor, hook: HookPoint):
        self.forward_cache.cache_dict[hook.name] = hook_point_out
        return hook_point_out

    def fwd_hooks(self) -> List[Tuple[str, Callable]]:
        return [(n, self.activation_mask_hook) for n in self.mask_logits_names] + [
            (n, self.caching_hook) for n in self.forward_cache_hook_points
        ]

    def with_fwd_hooks_and_new_ablation_cache(self, patch_data: PatchData) -> ContextManager[HookedTransformer]:
        self.calculate_and_store_ablation_cache(patch_data)
        return self.with_fwd_hooks()

    def with_fwd_hooks(self) -> ContextManager[HookedTransformer]:
        return self.model.hooks(self.fwd_hooks())

    def freeze_weights(self):
        for p in self.model.parameters():
            p.requires_grad = False

    def num_edges(self):
        values = []
        for name, mask in self._mask_parameter_dict.items():
            mask_value = self.sample_mask(name)
            values.extend(mask_value.flatten().tolist())
        values = torch.tensor(values)
        return (values > 0.5).sum().item()

    def num_params(self):
        return sum(p.numel() for p in self.mask_parameter_list)

    def get_edge_level_correspondence_from_masks(self, use_pos_embed: bool = None) -> TLACDCCorrespondence:
        if use_pos_embed is None:
            use_pos_embed = self.use_pos_embed
        corr = TLACDCCorrespondence.setup_from_model(self.model, use_pos_embed=use_pos_embed)

        # Define edges
        def indexes(name):
            if "mlp" in name or "resid" in name or "embed" in name or name == "blocks.0.hook_resid_pre":
                return [TorchIndex((None,))]
            return [TorchIndex((None, None, i)) for i in range(self.n_heads)]

        for child, mask_logits in self._mask_parameter_dict.items():
            # Sample mask for this child
            sampled_mask = self.sample_mask(child)
            # print(f"sampled mask for {child} has shape {sampled_mask.shape}")
            mask_row = 0
            for parent in self.hook_point_to_parents[child]:
                for parent_index in indexes(parent):
                    for mask_col, child_index in enumerate(indexes(child)):
                        # print(f"Setting edge {child} {child_index} <- {parent} {parent_index} to {mask_row, mask_col}")
                        # print(f"={sampled_mask[mask_row][mask_col]}")
                        mask_value = sampled_mask[mask_row][mask_col].item()
                        edge = corr.edges[child][child_index][parent][parent_index]
                        edge.present = mask_value > 0.5
                        edge.effect_size = mask_value
                    mask_row += 1

        # Delete a node's incoming edges if it has no outgoing edges and is not the output
        def get_nodes_with_out_edges(corr):
            nodes_with_out_edges = set()
            for (
                receiver_name,
                receiver_index,
                sender_name,
                sender_index,
            ), edge in corr.all_edges().items():
                nodes_with_out_edges.add(sender_name)
            return nodes_with_out_edges

        nodes_to_keep = get_nodes_with_out_edges(corr) | {f"blocks.{self.model.cfg.n_layers - 1}.hook_resid_post"}
        for (
            receiver_name,
            receiver_index,
            sender_name,
            sender_index,
        ), edge in corr.all_edges().items():
            if receiver_name not in nodes_to_keep:
                edge.present = False
        return corr

    def proportion_of_binary_scores(self) -> float:
        """How many of the scores are binary, i.e. 0 or 1
        (after going through the sigmoid with fp32 precision loss)
        """
        binary_count = 0
        total_count = 0

        for mask_name in self.mask_logits_names:
            mask = self.sample_mask(mask_name)
            for v in mask.view(-1):
                total_count += 1
                if v == 0 or v == 1:
                    binary_count += 1
        return binary_count / total_count
