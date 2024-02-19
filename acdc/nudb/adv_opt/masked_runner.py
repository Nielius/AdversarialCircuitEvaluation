import itertools
from contextlib import contextmanager
from functools import cached_property
from typing import Generator

import torch
from jaxtyping import Float, Integer, Num
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint

from acdc.TLACDCEdge import Edge, HookPointName, IndexedHookPointName
from subnetwork_probing.masked_transformer import EdgeLevelMaskedTransformer


class MaskedRunner:
    """
    A class to run a forward pass on a HookedTransformer, with some edges disabled.

    This class is intended to be mostly stateless."""

    masked_transformer: EdgeLevelMaskedTransformer

    _parent_index_per_child: dict[tuple[HookPointName, IndexedHookPointName], int]
    _indexed_parents_per_child: dict[HookPointName, list[IndexedHookPointName]]

    def __init__(self, model: HookedTransformer):
        assert (
            model.cfg.positional_embedding_type in {"standard"}
        ), "This is a temporary check; I don't know what values are possible here and what to do with them (in terms of whether or not they're using pos embed)"
        self.masked_transformer = EdgeLevelMaskedTransformer(
            model=model, use_pos_embed=model.cfg.positional_embedding_type == "standard"
        )
        self.masked_transformer.freeze_weights()
        self._freeze_all_masks()
        self._set_all_masks_to_pos_infty()
        self._set_up_parent_index_per_child()

    def _set_up_parent_index_per_child(self):
        # For every child, every possible parent in the edge has an index; this is the index of the mask in the list of masks for that child.
        # This sets up a look-up table for that.
        self._parent_index_per_child = {}
        self._indexed_parents_per_child = {}

        for child, all_parents in self.masked_transformer.parent_node_names.items():
            # we expand the list of all_parents into a list of all indexed parents, so that we can give
            # each of an index
            all_indexed_parents: list[IndexedHookPointName] = list(
                itertools.chain.from_iterable(
                    IndexedHookPointName.list_from_hook_point(name, self.masked_transformer.n_heads)
                    for name in all_parents
                )
            )
            self._indexed_parents_per_child[child] = all_indexed_parents
            for index, indexed_parent in enumerate(all_indexed_parents):
                self._parent_index_per_child[(child, indexed_parent)] = index

    def _freeze_all_masks(self):
        """In the MaskedTransformer, every mask is a parameter. In this case, however,
        we only want to run the model with fixed masks, so we freeze all the masks."""
        for value in self.masked_transformer._mask_logits_dict.values():
            value.requires_grad = False

    def _set_all_masks_to_pos_infty(self):
        for parameter in self.masked_transformer._mask_logits_dict.values():
            parameter.data.fill_(float("inf"))

    def _set_mask_for_edge(self, child: IndexedHookPointName, parent: IndexedHookPointName, value: float) -> None:
        parent_index = self._parent_index_per_child[(child.hook_name, parent)]
        # self._mask_logits_dict is dict[HookPointName of child, Num[torch.nn.Parameter, "parent (IndexedHookPoint), TorchIndex of child"]
        # todo: I think child.index.as_index[-1] shows that we're not using the right abstraction here; or maybe it doesn't?
        self.masked_transformer._mask_logits_dict[child.hook_name][parent_index][child.index.as_index[-1]] = value

    @cached_property
    def all_ablatable_edges(self) -> set[Edge]:
        return {
            Edge(child=indexed_child, parent=indexed_parent)
            for child, all_indexed_parents in self._indexed_parents_per_child.items()
            for indexed_child in IndexedHookPointName.list_from_hook_point(child, self.masked_transformer.n_heads)
            for indexed_parent in all_indexed_parents
        }

    @contextmanager
    def with_ablated_edges(
        self, patch_input: Num[torch.Tensor, "batch pos"], edges_to_ablate: list[Edge]
    ) -> Generator[HookedTransformer, None, None]:
        for edge in edges_to_ablate:
            assert edge in self.all_ablatable_edges  # safety check
            self._set_mask_for_edge(edge.child, edge.parent, float("-inf"))

        try:
            with self.masked_transformer.with_fwd_hooks_and_new_ablation_cache(patch_data=patch_input) as hooked_model:
                yield hooked_model

        finally:
            for edge in edges_to_ablate:
                self._set_mask_for_edge(
                    edge.child, edge.parent, float("inf")
                )  # this class is not intended to keep state

    def run(
        self,
        input: Num[torch.Tensor, "batch pos"],
        patch_input: Num[torch.Tensor, "batch pos"],
        edges_to_ablate: list[Edge],
    ) -> Num[torch.Tensor, "batch pos vocab"]:
        with self.with_ablated_edges(patch_input=patch_input, edges_to_ablate=edges_to_ablate) as hooked_model:
            return hooked_model(input)

    def run_with_linear_combination(
        self,
        input_embedded: Float[torch.Tensor, "batch pos"],
        dummy_input: Integer[torch.Tensor, "batch pos"],
        coefficients: Num[torch.Tensor, " batch"],
        patch_input: Num[torch.Tensor, "batch pos"],
        edges_to_ablate: list[Edge],
    ) -> Float[torch.Tensor, "1 pos vocab"]:
        """'input_embedded' should be the input after the embedding layer."""

        def replace_embedding_with_convex_combination_hook(
            hook_point_out: torch.Tensor, hook: HookPoint, verbose=False
        ) -> Float[torch.Tensor, "1 pos d_resid"]:
            # TODO: this probably be a convex combination of the embeddings, and also
            # with a softmax
            convex_combination = torch.einsum(
                "b, b p d -> p d",
                [torch.nn.functional.softmax(coefficients), input_embedded],
            ).unsqueeze(dim=0)
            return convex_combination

        # self.masked_transformer.init_ablation_cache(ablation=) <--- probably want to do something like this
        with self.with_ablated_edges(patch_input=patch_input, edges_to_ablate=edges_to_ablate) as hooked_transformer:
            assert isinstance(hooked_transformer, HookedTransformer)  # help PyCharm
            with hooked_transformer.hooks(
                fwd_hooks=[("hook_embed", replace_embedding_with_convex_combination_hook)]
            ) as hooked_with_linear_combination:
                return hooked_with_linear_combination(dummy_input)
