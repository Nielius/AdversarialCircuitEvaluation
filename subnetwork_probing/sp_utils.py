import collections
import math
from typing import Callable, ContextManager, Dict, List, Optional, Tuple

import torch
import wandb
from einops import rearrange
from torch.utils.checkpoint import checkpoint
from transformer_lens import HookedTransformer
from transformer_lens.ActivationCache import ActivationCache
from transformer_lens.hook_points import HookPoint

from acdc.TLACDCCorrespondence import TLACDCCorrespondence
from acdc.TLACDCEdge import EdgeType, TorchIndex
from acdc.TLACDCInterpNode import TLACDCInterpNode
from acdc.acdc_utils import get_edge_stats, get_node_stats


def set_ground_truth_edges(canonical_circuit_subgraph: TLACDCCorrespondence, ground_truth_set: set):
    for (
        receiver_name,
        receiver_index,
        sender_name,
        sender_index,
    ), edge in canonical_circuit_subgraph.all_edges().items():
        key = (receiver_name, receiver_index.hashable_tuple, sender_name, sender_index.hashable_tuple)
        edge.present = key in ground_truth_set


def print_stats(recovered_corr, ground_truth_subgraph, do_print=True):
    """
    False positive = present in recovered_corr but not in ground_truth_set
    False negative = present in ground_truth_set but not in recovered_corr
    """
    # diff = set(recovered_corr.all_edges().keys()) - set(ground_truth_subgraph.all_edges().keys())
    # if diff: print(f"{len(diff)} key mismatches: {diff}")
    stats = get_node_stats(ground_truth=ground_truth_subgraph, recovered=recovered_corr)
    node_tpr = stats["true positive"] / (stats["true positive"] + stats["false negative"])
    node_fpr = stats["false positive"] / (stats["false positive"] + stats["true negative"])
    if do_print:
        print(f"Node TPR: {node_tpr:.3f}. Node FPR: {node_fpr:.3f}")

    stats = get_edge_stats(ground_truth=ground_truth_subgraph, recovered=recovered_corr)
    edge_tpr = stats["true positive"] / (stats["true positive"] + stats["false negative"])
    edge_fpr = stats["false positive"] / (stats["false positive"] + stats["true negative"])
    if do_print:
        print(f"Edge TPR: {edge_tpr:.3f}. Edge FPR: {edge_fpr:.3f}")

    return {
        "node_tpr": node_tpr,
        "node_fpr": node_fpr,
        "edge_tpr": edge_tpr,
        "edge_fpr": edge_fpr,
    }


def iterative_correspondence_from_mask(
    model: HookedTransformer,
    nodes_to_mask: List[TLACDCInterpNode],  # Can be empty
    use_pos_embed: bool = False,
    corr: Optional[TLACDCCorrespondence] = None,
    head_parents: Optional[List] = None,
) -> Tuple[TLACDCCorrespondence, List]:
    """Given corr has some nodes masked, also mask the nodes_to_mask"""

    assert (corr is None) == (
        head_parents is None
    ), "Ensure we're either masking from scratch or we provide details on `head_parents`"

    if corr is None:
        corr = TLACDCCorrespondence.setup_from_model(model, use_pos_embed=use_pos_embed)
    if head_parents is None:
        head_parents = collections.defaultdict(lambda: 0)

    additional_nodes_to_mask = []

    for node in nodes_to_mask:
        additional_nodes_to_mask.append(
            TLACDCInterpNode(node.name.replace(".attn.", ".") + "_input", node.index, EdgeType.ADDITION)
        )

        if node.name.endswith("_q") or node.name.endswith("_k") or node.name.endswith("_v"):
            child_name = node.name.replace("_q", "_result").replace("_k", "_result").replace("_v", "_result")
            head_parents[(child_name, node.index)] += 1

            if head_parents[(child_name, node.index)] == 3:
                additional_nodes_to_mask.append(TLACDCInterpNode(child_name, node.index, EdgeType.PLACEHOLDER))

            # Forgot to add these in earlier versions of Subnetwork Probing, and so the edge counts were inflated
            additional_nodes_to_mask.append(TLACDCInterpNode(child_name + "_input", node.index, EdgeType.ADDITION))

        if node.name.endswith(("mlp_in", "resid_mid")):
            additional_nodes_to_mask.append(
                TLACDCInterpNode(
                    node.name.replace("resid_mid", "mlp_out").replace("mlp_in", "mlp_out"),
                    node.index,
                    EdgeType.DIRECT_COMPUTATION,
                )
            )

    assert all(
        [v <= 3 for v in head_parents.values()]
    ), "We should have at most three parents (Q, K and V, connected via placeholders)"

    for node in nodes_to_mask + additional_nodes_to_mask:
        # Mark edges where this is child as not present
        rest2 = corr.edges[node.name][node.index]
        for rest3 in rest2.values():
            for edge in rest3.values():
                edge.present = False

        # Mark edges where this is parent as not present
        for rest1 in corr.edges.values():
            for rest2 in rest1.values():
                if node.name in rest2 and node.index in rest2[node.name]:
                    rest2[node.name][node.index].present = False

    return corr, head_parents


class MaskedTransformer(torch.nn.Module):
    """
    A wrapper around HookedTransformer that allows edge-level subnetwork probing.

    There are two sets of hooks:
    - `activation_mask_hook`s change the input to a node. The input to a node is the sum
      of several residual stream terms; ablated edges are looked up from `ablation_cache`
      and non-ablated edges from `forward_cache`, then the sum is taken.
    - `caching_hook`s save the output of a node to `forward_cache` for use in later layers.
    """

    model: HookedTransformer
    ablation_cache: ActivationCache
    forward_cache: ActivationCache
    mask_logits: torch.nn.ParameterList
    _mask_logits_dict: Dict[str, torch.nn.Parameter]

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
        self.mask_logits = torch.nn.ParameterList()
        self._mask_logits_dict = {}
        self.no_ablate = no_ablate
        if no_ablate:
            print("WARNING: no_ablate is True, this is for testing only")
        self.device = self.model.parameters().__next__().device
        self.use_pos_embed = use_pos_embed
        self.verbose = verbose

        # Stores the cache keys that correspond to each mask,
        # e.g. ...1.hook_mlp_in -> ["blocks.0.attn.hook_result", "blocks.0.hook_mlp_out", "blocks.1.attn.hook_result"]
        # Logits are attention in-edges, then MLP in-edges
        self.parent_node_names: Dict[str, list[str]] = {}

        self.ablation_cache = ActivationCache({}, self.model)
        self.forward_cache = ActivationCache({}, self.model)
        self.a_cache_tensor = None
        self.f_cache_tensor = None
        self.cache_indices_dict = {}  # Converts a hook name to an integer representing how far to index?
        # Hyperparameters
        self.beta = beta
        self.gamma = gamma
        self.zeta = zeta
        self.mask_init_p = mask_init_p

        # Copied from subnetwork probing code. Similar to log odds (but not the same)
        p = (self.mask_init_p - self.gamma) / (self.zeta - self.gamma)
        self.mask_init_constant = math.log(p / (1 - p))

        model.cfg.use_hook_mlp_in = True  # We need to hook the MLP input to do subnetwork probing

        self.embeds = ["hook_embed", "hook_pos_embed"] if self.use_pos_embed else ["blocks.0.hook_resid_pre"]

        self.forward_cache_names = self.embeds[:]
        self.cache_indices_dict = {name: (i, i + 1) for i, name in enumerate(self.forward_cache_names)}
        self.n_units_so_far = len(self.embeds)
        # Add mask logits for ablation cache
        # Mask logits have a variable dimension depending on the number of in-edges (increases with layer)
        for layer_i in range(model.cfg.n_layers):
            # QKV: in-edges from all previous layers
            for q_k_v in ["q", "k", "v"]:
                self._setup_mask_logits(mask_name=f"blocks.{layer_i}.hook_{q_k_v}_input", out_dim=self.n_heads)

            self.forward_cache_names.append(f"blocks.{layer_i}.attn.hook_result")
            self.cache_indices_dict[f"blocks.{layer_i}.attn.hook_result"] = (
                self.n_units_so_far,
                self.n_units_so_far + self.n_heads,
            )
            self.n_units_so_far += self.n_heads

            # MLP: in-edges from all previous layers and current layer's attention heads
            if not model.cfg.attn_only:
                self._setup_mask_logits(mask_name=f"blocks.{layer_i}.hook_mlp_in", out_dim=1)

                self.forward_cache_names.append(f"blocks.{layer_i}.hook_mlp_out")
                self.cache_indices_dict[f"blocks.{layer_i}.hook_mlp_out"] = (
                    self.n_units_so_far,
                    self.n_units_so_far + 1,
                )
                self.n_units_so_far += 1

        self._setup_mask_logits(mask_name=f"blocks.{model.cfg.n_layers - 1}.hook_resid_post", out_dim=1)

        print(self.forward_cache_names, self.parent_node_names)
        for ckl in self.parent_node_names.values():
            for name in ckl:
                assert name in self.forward_cache_names, f"{name} not in forward cache names"

    @property
    def mask_logits_names(self):
        return self._mask_logits_dict.keys()

    def _setup_mask_logits(self, mask_name, out_dim):
        """
        Adds a mask logit for the given mask name and parent nodes
        Parent nodes are (attention, MLP)
        """
        in_dim = self.n_units_so_far
        self.parent_node_names[mask_name] = self.forward_cache_names[:]
        self.mask_logits.append(
            torch.nn.Parameter(torch.full((in_dim, out_dim), self.mask_init_constant, device=self.device))
        )
        self._mask_logits_dict[mask_name] = self.mask_logits[-1]

    def sample_mask(self, mask_name) -> torch.Tensor:
        """Samples a binary-ish mask from the mask_scores for the particular `mask_name` activation"""
        mask_scores = self._mask_logits_dict[mask_name]
        uniform_sample = torch.zeros_like(mask_scores, requires_grad=False).uniform_().clamp_(0.0001, 0.9999)
        s = torch.sigmoid((uniform_sample.log() - (1 - uniform_sample).log() + mask_scores) / self.beta)
        s_bar = s * (self.zeta - self.gamma) + self.gamma
        mask = s_bar.clamp(min=0.0, max=1.0)
        # print(f"Displaying grad tree of {mask_name}")
        # def get_grad_tree(f):
        #     if f is None: return str(f)
        #     return f"{f} -> ({', '.join(get_grad_tree(tup[0]) for tup in f.next_functions)})"
        # a = mask.grad_fn
        # print(get_grad_tree(a))

        return mask

    def regularization_loss(self) -> torch.Tensor:
        center = self.beta * math.log(-self.gamma / self.zeta)
        per_parameter_loss = [torch.sigmoid(scores - center).mean() for scores in self.mask_logits]
        return torch.mean(torch.stack(per_parameter_loss))

    @staticmethod
    def make_4d(x):
        if x.ndim == 3:
            return x.unsqueeze(2)
        return x

    def do_zero_caching(self):
        """Caches zero for every possible mask point."""
        patch_data = torch.zeros((1, 1), device=self.device, dtype=torch.int64)  # batch pos
        self.do_random_resample_caching(patch_data)
        self.ablation_cache.cache_dict = {
            name: torch.zeros_like(scores) for name, scores in self.ablation_cache.cache_dict.items()
        }
        # self.a_cache_tensor = torch.cat([self.make_4d(self.ablation_cache[name]) for name in self.forward_cache_names], dim=2)
        # self.a_cache_tensor.requires_grad_(False)

    def do_random_resample_caching(self, patch_data) -> None:
        # Only cache the tensors needed to fill the masked out positions
        with torch.no_grad():
            model_out, self.ablation_cache = self.model.run_with_cache(
                patch_data, names_filter=lambda name: name in self.forward_cache_names, return_cache_object=True
            )
            # self.a_cache_tensor = torch.cat([self.make_4d(self.ablation_cache[name]) for name in self.forward_cache_names], dim=2)
            # self.a_cache_tensor.requires_grad_(False)

    def get_activation_values(self, names, cache: ActivationCache):
        """
        Returns a single tensor of the mask values used for a given hook.
        Attention is shape batch, seq, heads, head_size while MLP out is batch, seq, d_model
        so we need to reshape things to match
        """
        result = []
        for name in names:
            value = cache[name]  # b s n_heads d, or b s d
            if value.ndim == 3:
                value = value.unsqueeze(2)  # b s 1 d
            result.append(value)
        return torch.cat(result, dim=2)

    def compute_weighted_values(self, hook: HookPoint):
        names = self.parent_node_names[hook.name]
        a_values = self.get_activation_values(names, self.ablation_cache)  # b s i d
        f_values = self.get_activation_values(names, self.forward_cache)  # b s i d
        mask = self.sample_mask(hook.name)  # in_edges, nodes_per_mask, ...

        weighted_a_values = torch.einsum("b s i d, i o -> b s o d", a_values, 1 - mask)
        weighted_f_values = torch.einsum("b s i d, i o -> b s o d", f_values, mask)
        return weighted_a_values + weighted_f_values

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

        block_num = int(hook.name.split(".")[1])
        for layer in self.model.blocks[: block_num if "mlp" in hook.name else 1]:
            out += layer.attn.b_O

        if self.no_ablate and not torch.allclose(hook_point_out, out):
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
        start, end = self.cache_indices_dict[hook.name]
        if self.f_cache_tensor is None:
            batch, seq, d_model = hook_point_out.shape
            self.f_cache_tensor = torch.zeros((batch, seq, self.n_units_so_far, d_model), device=self.device)
        self.f_cache_tensor[:, :, start:end, :] = self.make_4d(hook_point_out)
        return hook_point_out

    def fwd_hooks(self) -> List[Tuple[str, Callable]]:
        return [(n, self.activation_mask_hook) for n in self.mask_logits_names] + [
            (n, self.caching_hook) for n in self.forward_cache_names
        ]

    def with_fwd_hooks_and_new_cache(
        self, ablation="resample", ablation_data=None
    ) -> ContextManager[HookedTransformer]:
        assert ablation in ["zero", "resample"]
        if ablation == "zero":
            self.do_zero_caching()
        else:
            assert ablation_data is not None
            self.do_random_resample_caching(ablation_data)
        return self.model.hooks(self.fwd_hooks())

    def freeze_weights(self):
        for p in self.model.parameters():
            p.requires_grad = False

    def num_edges(self):
        values = []
        for name, mask in self._mask_logits_dict.items():
            mask_value = self.sample_mask(name)
            values.extend(mask_value.flatten().tolist())
        values = torch.tensor(values)
        return (values > 0.5).sum().item()

    def num_params(self):
        return sum(p.numel() for p in self.mask_logits)


def edge_level_corr(masked_model: MaskedTransformer, use_pos_embed: bool = None) -> TLACDCCorrespondence:
    if use_pos_embed is None:
        use_pos_embed = masked_model.use_pos_embed
    corr = TLACDCCorrespondence.setup_from_model(masked_model.model, use_pos_embed=use_pos_embed)

    # Define edges
    def indexes(name):
        if "mlp" in name or "resid" in name or "embed" in name or name == "blocks.0.hook_resid_pre":
            return [TorchIndex((None,))]
        return [TorchIndex((None, None, i)) for i in range(masked_model.n_heads)]

    for child, mask_logits in masked_model._mask_logits_dict.items():
        # Sample mask for this child
        sampled_mask = masked_model.sample_mask(child)
        # print(f"sampled mask for {child} has shape {sampled_mask.shape}")
        mask_row = 0
        for parent in masked_model.parent_node_names[child]:
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
        for (receiver_name, receiver_index, sender_name, sender_index), edge in corr.all_edges().items():
            nodes_with_out_edges.add(sender_name)
        return nodes_with_out_edges

    nodes_to_keep = get_nodes_with_out_edges(corr) | {f"blocks.{masked_model.model.cfg.n_layers - 1}.hook_resid_post"}
    for (receiver_name, receiver_index, sender_name, sender_index), edge in corr.all_edges().items():
        if receiver_name not in nodes_to_keep:
            edge.present = False
    return corr


def log_plotly_bar_chart(x: List[str], y: List[float]) -> None:
    import plotly.graph_objects as go

    fig = go.Figure(data=[go.Bar(x=x, y=y)])
    wandb.log({"mask_scores": fig})


def visualize_mask(masked_model: MaskedTransformer) -> tuple[int, list[TLACDCInterpNode]]:
    # This is bad code, shouldn't combine visualizing and getting the nodes to mask
    number_of_heads = masked_model.model.cfg.n_heads
    number_of_layers = masked_model.model.cfg.n_layers
    node_name_list = []
    mask_scores_for_names = []
    total_nodes = 0
    nodes_to_mask: list[TLACDCInterpNode] = []
    for layer_index in range(number_of_layers):
        for head_index in range(number_of_heads):
            for q_k_v in ["q", "k", "v"]:
                total_nodes += 1
                node_name = f"blocks.{layer_index}.attn.hook_{q_k_v}"
                mask_sample = masked_model.sample_mask(node_name)[head_index].cpu().item()

                node_name_with_index = f"{node_name}[{head_index}]"
                node_name_list.append(node_name_with_index)
                node = TLACDCInterpNode(
                    node_name, TorchIndex((None, None, head_index)), incoming_edge_type=EdgeType.ADDITION
                )

                mask_scores_for_names.append(mask_sample)
                if mask_sample < 0.5:
                    nodes_to_mask.append(node)

        # MLPs
        # This is actually fairly wrong for getting the exact nodes and edges we keep in the circuit but in the `filter_nodes` function
        # used in post-processing (in roc_plot_generator.py we process hook_resid_mid/mlp_in and mlp_out hooks together properly) we iron
        # these errors so that plots are correct
        node_name = f"blocks.{layer_index}.hook_mlp_out"
        mask_sample = masked_model.sample_mask(node_name).cpu().item()
        mask_scores_for_names.append(mask_sample)
        node_name_list.append(node_name)

        for node_name, edge_type in [
            (f"blocks.{layer_index}.hook_mlp_out", EdgeType.PLACEHOLDER),
            (f"blocks.{layer_index}.hook_resid_mid", EdgeType.ADDITION),
        ]:
            node = TLACDCInterpNode(node_name, TorchIndex([None]), incoming_edge_type=edge_type)
            total_nodes += 1

            if mask_sample < 0.5:
                nodes_to_mask.append(node)

    # assert len(mask_scores_for_names) == 3 * number_of_heads * number_of_layers
    log_plotly_bar_chart(x=node_name_list, y=mask_scores_for_names)
    node_count = total_nodes - len(nodes_to_mask)
    return node_count, nodes_to_mask
