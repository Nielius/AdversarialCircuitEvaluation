import logging
import math
from contextlib import contextmanager
from enum import Enum
from typing import (
    Callable,
    ContextManager,
    Iterable,
    Iterator,
    Sequence,
    TypeAlias,
    Union,
    cast,
)

import torch
from einops import rearrange
from jaxtyping import Float, Num
from transformer_lens import ActivationCache, HookedTransformer
from transformer_lens.hook_points import HookPoint, NamesFilter
from transformer_lens.HookedTransformer import Loss

from acdc.acdc_utils import get_present_nodes
from acdc.TLACDCCorrespondence import TLACDCCorrespondence
from acdc.TLACDCEdge import HookPointName, TorchIndex

logger = logging.getLogger(__name__)

PatchData: TypeAlias = Num[torch.Tensor, "batch pos"] | None  # use None if you want zero ablation


class CircuitStartingPointType(str, Enum):
    """We have two conventions for where to start the circuit: either at hook_embed and hook_pos_embed, or at
    hook_resid_pre.

    In older pieces of code this concept might also be referred to as 'use_pos_embed: bool' (where True corresponds
    to CircuitStartingPointType.POS_EMBED).
    """

    POS_EMBED = "pos_embed"  # uses hook_embed and hook_pos_embed as starting point
    RESID_PRE = "resid_pre"  # uses blocks.0.hook_resid_pre as starting point


def create_mask_parameters_and_forward_cache_hook_points(
    circuit_start_type: CircuitStartingPointType,
    num_heads: int,
    num_layers: int,
    device: str | torch.device,
    mask_init_constant: float,
    attn_only: bool,
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
            torch.full(
                (num_output_units_so_far, num_instances),
                mask_init_constant,
                device=device,
            )
        )
        mask_parameter_list.append(new_mask_parameter)
        mask_parameter_dict[mask_name] = new_mask_parameter  # pyright: ignore reportArgumentType  # seems to be an issue with pyright or with torch.nn.Parameter?

    match circuit_start_type:
        case CircuitStartingPointType.POS_EMBED:
            starting_points = ["hook_embed", "hook_pos_embed"]
        case CircuitStartingPointType.RESID_PRE:
            starting_points = ["blocks.0.hook_resid_pre"]
        case _:
            raise ValueError(f"Unknown circuit_start_type: {circuit_start_type}")

    for embedding_hook_point in starting_points:
        setup_output_hook_point(embedding_hook_point, 1)

    # Add mask logits for ablation cache
    # Mask logits have a variable dimension depending on the number of in-edges (increases with layer)
    for layer_i in range(num_layers):
        for q_k_v in ["q", "k", "v"]:
            setup_input_hook_point(
                mask_name=f"blocks.{layer_i}.hook_{q_k_v}_input",
                num_instances=num_heads,
            )

        setup_output_hook_point(f"blocks.{layer_i}.attn.hook_result", num_heads)

        if not attn_only:
            setup_input_hook_point(mask_name=f"blocks.{layer_i}.hook_mlp_in", num_instances=1)
            setup_output_hook_point(f"blocks.{layer_i}.hook_mlp_out", num_instances=1)

    # why does this get a mask? isn't it pointless to mask this?
    setup_input_hook_point(mask_name=f"blocks.{num_layers - 1}.hook_resid_post", num_instances=1)

    return (
        ordered_forward_cache_hook_points,
        hook_point_to_parents,
        mask_parameter_list,
        mask_parameter_dict,
    )


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
        starting_point_type: CircuitStartingPointType = CircuitStartingPointType.POS_EMBED,
        no_ablate=False,
        verbose=False,
    ):
        """
        - 'use_pos_embed': if set to True, create masks for edges from 'hook_embed' and 'hook_pos_embed'; othererwise,
            create masks for edges from 'blocks.0.hook_resid_pre'.
        """
        super().__init__()

        self.model = model
        self.n_heads = model.cfg.n_heads
        self.n_mlp = 0 if model.cfg.attn_only else 1
        self.no_ablate = no_ablate
        if no_ablate:
            print("WARNING: no_ablate is True, this is for testing only")
        self.device = self.model.parameters().__next__().device
        self.starting_point_type = starting_point_type
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
            circuit_start_type=self.starting_point_type,
            num_heads=self.n_heads,
            num_layers=model.cfg.n_layers,
            device=self.device,
            mask_init_constant=math.log(p / (1 - p)),
            attn_only=model.cfg.attn_only,
        )

    @property
    def mask_parameter_names(self) -> Iterable[str]:
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

    def _calculate_and_store_resampling_ablation_cache(
        self,
        patch_data: Num[torch.Tensor, "batch pos"],
        retain_cache_gradients: bool = False,
    ) -> None:
        # Only cache the tensors needed to fill the masked out positions
        if not retain_cache_gradients:
            with torch.no_grad():
                model_out, self.ablation_cache = self.model.run_with_cache(
                    patch_data,
                    names_filter=lambda name: name in self.forward_cache_hook_points,
                    return_cache_object=True,
                )
        else:
            model_out, self.ablation_cache = self.run_with_attached_cache(
                patch_data,
                names_filter=lambda name: name in self.forward_cache_hook_points,
            )

    def run_with_attached_cache(
        self, *model_args, names_filter: NamesFilter = None
    ) -> tuple[
        Union[
            None,
            Float[torch.Tensor, "batch pos d_vocab"],
            Loss,
            tuple[Float[torch.Tensor, "batch pos d_vocab"], Loss],
        ],
        ActivationCache,
    ]:
        """An adaptation of HookedTransformer.run_with_cache that does not
        detach the tensors in the cache. This means we can calculate the gradient
        of the patch data from the cache."""
        cache_dict = {}

        def save_hook(tensor, hook):
            # This is the essential difference with HookedTransformer.run_with_cache:
            # it uses a hook that detaches the tensor rather than just storing it
            cache_dict[hook.name] = tensor

        names_filter = self._convert_names_filter(names_filter)
        fwd_hooks = cast(
            list[tuple[str | Callable, Callable]],
            [(name, save_hook) for name in self.model.hook_dict.keys() if names_filter(name)],
        )

        with self.hooks(fwd_hooks=fwd_hooks) as runner_with_cache:
            out = runner_with_cache.model(*model_args)

        return out, ActivationCache(
            cache_dict=cache_dict,
            model=self,
        )

    @staticmethod
    def _convert_names_filter(names_filter: NamesFilter) -> Callable[[str], bool]:
        """This is extracted from HookedRootModule.get_caching_hooks."""
        if names_filter is None:
            names_filter = lambda name: True  # noqa: E731
        elif isinstance(names_filter, str):
            filter_str = names_filter
            names_filter = lambda name: name == filter_str  # noqa: E731
        elif isinstance(names_filter, Sequence):
            filter_list = names_filter
            names_filter = lambda name: name in filter_list  # noqa: E731

        return names_filter

    def calculate_and_store_ablation_cache(self, patch_data: PatchData, retain_cache_gradients: bool = True):
        """Use None for the patch data for zero ablation.

        If you want to calculate gradients on the patch data/the cache, set retain_cache_gradients=True.
        Otherwise set to False for performance."""
        if patch_data is None:
            self._calculate_and_store_zero_ablation_cache()
        else:
            self._calculate_and_store_resampling_ablation_cache(
                patch_data, retain_cache_gradients=retain_cache_gradients
            )

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

    def compute_weighted_values(self, hook: HookPoint) -> Float[torch.Tensor, "batch pos head_index d_resid"]:
        parent_names = self.hook_point_to_parents[hook.name]  # pyright: ignore # hook.name is not typed correctly
        ablation_values = self.get_activation_values(parent_names, self.ablation_cache)  # b s i d (i = parentindex)
        forward_values = self.get_activation_values(parent_names, self.forward_cache)  # b s i d
        mask = self.sample_mask(
            hook.name  # pyright: ignore # hook.name is not typed correctly
        )  # in_edges, nodes_per_mask, ...

        weighted_ablation_values = torch.einsum("b s i d, i o -> b s o d", ablation_values, 1 - mask)
        weighted_forward_values = torch.einsum("b s i d, i o -> b s o d", forward_values, mask)
        return weighted_ablation_values + weighted_forward_values

    def activation_mask_hook(self, hook_point_out: torch.Tensor, hook: HookPoint):
        """
        For edge-level SP, we discard the hook_point_out value and resum the residual stream.
        """
        if self.verbose:
            print(f"Doing ablation of {hook.name}")
            print(f"Using memory {torch.cuda.memory_allocated():_} bytes at hook start")
        is_attn = "mlp" not in hook.name and "resid_post" not in hook.name  # pyright: ignore # hook.name is not typed correctly

        # To trade off CPU against memory, you can use
        #    out = checkpoint(self.compute_weighted_values, hook, use_reentrant=False)
        # However, that messes with the backward pass, so I've disabled it for now.
        out = self.compute_weighted_values(hook)
        if not is_attn:
            out = rearrange(out, "b s 1 d -> b s d")

        # add back attention bias
        # Explanation: the attention bias is not part of the cached values (why not?), so we need to add it back here
        # we need to iterate over all attention layers that come before current layer
        current_block_index = int(hook.name.split(".")[1])  # pyright: ignore # hook.name is not typed correctly
        last_attention_block_index = (
            current_block_index + 1 if ("resid_post" in hook.name or "mlp" in hook.name) else current_block_index  # pyright: ignore # hook.name is not typed correctly
        )
        for layer in self.model.blocks[:last_attention_block_index]:  # pyright: ignore
            out += layer.attn.b_O

        if self.no_ablate and not torch.allclose(hook_point_out, out, atol=1e-4):
            print(f"Warning: hook_point_out and out are not close for {hook.name}")
            print(f"{hook_point_out.mean()=}, {out.mean()=}")

        if self.verbose:
            no_change = torch.allclose(hook_point_out, out)
            absdiff = (hook_point_out - out).abs().mean()
            print(f"Ablation hook {'did NOT' if no_change else 'DID'} change {hook.name} by {absdiff:.3f}")
        torch.cuda.empty_cache()
        if self.verbose:
            print(f"Using memory {torch.cuda.memory_allocated():_} bytes after clearing cache")
        return out

    def caching_hook(self, hook_point_out: torch.Tensor, hook: HookPoint):
        assert hook.name is not None
        self.forward_cache.cache_dict[hook.name] = hook_point_out
        return hook_point_out

    def fwd_hooks(self) -> list[tuple[str | Callable, Callable]]:
        return cast(
            list[tuple[str | Callable, Callable]],
            [(hook_point, self.activation_mask_hook) for hook_point in self.mask_parameter_names]
            + [(hook_point, self.caching_hook) for hook_point in self.forward_cache_hook_points],
        )

    def with_fwd_hooks(self) -> ContextManager[HookedTransformer]:
        return self.model.hooks(self.fwd_hooks())

    def with_fwd_hooks_and_new_ablation_cache(self, patch_data: PatchData) -> ContextManager[HookedTransformer]:
        self.calculate_and_store_ablation_cache(patch_data, retain_cache_gradients=False)
        return self.with_fwd_hooks()

    @contextmanager
    def hooks(
        self,
        fwd_hooks: list[tuple[str | Callable, Callable]] | None = None,
        bwd_hooks: list[tuple[str | Callable, Callable]] | None = None,
        reset_hooks_end: bool = True,
        clear_contexts: bool = False,
    ) -> Iterator["EdgeLevelMaskedTransformer"]:
        """Imitates the 'hooks' context manager in HookedTransformer."""
        with self.model.hooks(
            fwd_hooks=fwd_hooks or [],
            bwd_hooks=bwd_hooks or [],
            reset_hooks_end=reset_hooks_end,
            clear_contexts=clear_contexts,
        ):
            # the with-statement above updates the hooks in self.model
            # so we can simply yield self
            yield self

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

    def get_edge_level_correspondence_from_masks(self, use_pos_embed: bool | None = None) -> TLACDCCorrespondence:
        if use_pos_embed is None:
            use_pos_embed = self.starting_point_type == CircuitStartingPointType.POS_EMBED
        corr = TLACDCCorrespondence.setup_from_model(self.model, use_pos_embed=use_pos_embed)

        # Define edges
        def indexes(name):
            if "mlp" in name or "resid" in name or "embed" in name or name == "blocks.0.hook_resid_pre":
                return [TorchIndex((None,))]
            return [TorchIndex((None, None, i)) for i in range(self.n_heads)]

        # Step 1: Set/unset edges based on samples from mask

        for child, mask_logits in self._mask_parameter_dict.items():
            # Sample mask for this child
            sampled_mask = self.sample_mask(child)
            mask_row = 0
            for parent in self.hook_point_to_parents[child]:
                for parent_index in indexes(parent):
                    for mask_col, child_index in enumerate(indexes(child)):
                        mask_value = sampled_mask[mask_row][mask_col].item()
                        edge = corr.edges[child][child_index][parent][parent_index]
                        edge.present = mask_value > 0.5
                        edge.effect_size = mask_value
                    mask_row += 1
        if self.verbose:
            print("\n-----\nNumber of present edges (num_edges):", self.num_edges())
            present_nodes, all_nodes = get_present_nodes(corr)
            print(f"Total number of nodes in edge_dict: {len(all_nodes)}")
            print(f"Number of nodes present after sampling: {len(present_nodes)}")

        # Step 2: Now we need to deal with edges that are not present in the mask

        def get_nodes_with_out_edges(corr: TLACDCCorrespondence):
            """Returns a set of nodes that have outgoing edges"""
            nodes_with_out_edges = set()
            for (
                receiver_name,
                receiver_index,
                sender_name,
                sender_index,
            ), edge in corr.edge_dict().items():
                if edge.present:  # only consider present edges
                    nodes_with_out_edges.add((sender_name, sender_index))
            return nodes_with_out_edges

        # Recursively remove edges to nodes that have no outgoing edges and is not the output node
        while True:
            edges_removed = 0
            receiver_nodes_to_keep = get_nodes_with_out_edges(corr)
            receiver_nodes_to_keep.add(
                (
                    f"blocks.{self.model.cfg.n_layers - 1}.hook_resid_post",
                    TorchIndex((None,)),
                )
            )
            for (
                receiver_name,
                receiver_index,
                sender_name,
                sender_index,
            ), edge in corr.edge_dict().items():
                if (receiver_name, receiver_index) not in receiver_nodes_to_keep:
                    if edge.present:  # this condition is required to eventually break the loop
                        if self.verbose:
                            print(
                                "Removing edge",
                                f"({receiver_name}, {receiver_index}) <- ({sender_name}, {sender_index})",
                                "because it has no outgoing edges",
                            )
                        edge.present = False
                        edges_removed += 1
            if edges_removed == 0:  # this means we have pruned all nodes
                break
        if self.verbose:
            print("\n\nremoved leaf nodes")
            present_nodes, all_nodes = get_present_nodes(corr)
            present_edges = sum([1 for edge in corr.edge_dict().values() if edge.present])
            print(f"Number of edges present after pruning leaf nodes: {present_edges}")
            print(f"Number of nodes present after pruning leaf nodes: {len(present_nodes)}")

        return corr

    def proportion_of_binary_scores(self) -> float:
        """How many of the scores are binary, i.e. 0 or 1
        (after going through the sigmoid with fp32 precision loss)
        """
        binary_count = 0
        total_count = 0

        for mask_name in self.mask_parameter_names:
            mask = self.sample_mask(mask_name)
            for v in mask.view(-1):
                total_count += 1
                if v == 0 or v == 1:
                    binary_count += 1
        return binary_count / total_count
