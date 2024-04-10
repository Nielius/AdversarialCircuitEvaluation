import itertools
from contextlib import contextmanager
from functools import cached_property
from typing import Callable, Iterator

import torch
from jaxtyping import Float, Integer, Num
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint

from acdc.TLACDCEdge import Edge, HookPointName, IndexedHookPointName
from subnetwork_probing.masked_transformer import CircuitStartingPointType, EdgeLevelMaskedTransformer


class MaskedRunner:
    """
    A class to run a forward pass on a HookedTransformer, with some edges disabled.

    This class is intended to be mostly stateless."""

    masked_transformer: EdgeLevelMaskedTransformer

    _parent_index_per_child: dict[tuple[HookPointName, IndexedHookPointName], int]
    _indexed_parents_per_child: dict[HookPointName, list[IndexedHookPointName]]

    def __init__(self, model: HookedTransformer, starting_point_type: CircuitStartingPointType):
        assert (
            model.cfg.positional_embedding_type in {"standard"}
        ), "This is a temporary check; I don't know what values are possible here and what to do with them (in terms of whether or not they're using pos embed)"
        self.masked_transformer = EdgeLevelMaskedTransformer(model=model, starting_point_type=starting_point_type)
        self.masked_transformer.freeze_weights()
        self._freeze_all_masks()
        self._set_all_masks_to_pos_infty()
        self._set_up_parent_index_per_child()

    def _set_up_parent_index_per_child(self):
        # For every child, every possible parent in the edge has an index; this is the index of the mask in the list of masks for that child.
        # This sets up a look-up table for that.
        self._parent_index_per_child = {}
        self._indexed_parents_per_child = {}

        for child, all_parents in self.masked_transformer.hook_point_to_parents.items():
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
        for value in self.masked_transformer._mask_parameter_dict.values():
            value.requires_grad = False

    def _set_all_masks_to_pos_infty(self):
        for parameter in self.masked_transformer._mask_parameter_dict.values():
            parameter.data.fill_(float("inf"))

    def _set_mask_for_edge(self, child: IndexedHookPointName, parent: IndexedHookPointName, value: float) -> None:
        parent_index = self._parent_index_per_child[(child.hook_name, parent)]
        # self._mask_logits_dict is dict[HookPointName of child, Num[torch.nn.Parameter, "parent (IndexedHookPoint), TorchIndex of child"]
        # todo: I think child.index.as_index[-1] shows that we're not using the right abstraction here; or maybe it doesn't?
        self.masked_transformer._mask_parameter_dict[child.hook_name][parent_index][child.index.as_index[-1]] = value  # pyright: ignore

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
        self,
        patch_input: Num[torch.Tensor, "batch pos"] | None,
        edges_to_ablate: list[Edge],
        retain_patch_gradient: bool = False,
    ) -> Iterator[HookedTransformer]:
        """If 'patch_input' is None, do not recalculate the ablation cache. This is useful if you're running the model
        with the same patch input as before, and you don't want to recalculate the ablation cache.
        It is also useful if you're adding hooks that get in the way of the ablation cache calculation.

        If 'patch_input' is None, 'retain_gradient_for_ablated_edges' is ignored.
        If 'patch_input' is not None, 'retain_gradient_for_ablated_edges' determines whether the gradient for the patch
        input is retained in the ablation cache."""
        for edge in edges_to_ablate:
            assert edge in self.all_ablatable_edges  # safety check
            self._set_mask_for_edge(edge.child, edge.parent, float("-inf"))

        try:
            if patch_input is not None:
                self.masked_transformer.calculate_and_store_ablation_cache(
                    patch_input, retain_cache_gradients=retain_patch_gradient
                )

            with self.masked_transformer.with_fwd_hooks() as hooked_model:
                yield hooked_model

        finally:
            for edge in edges_to_ablate:
                self._set_mask_for_edge(
                    edge.child, edge.parent, float("inf")
                )  # this class is not intended to keep state

    @contextmanager
    def hooks(
        self,
        fwd_hooks: list[tuple[str | Callable, Callable]] | None = None,
        bwd_hooks: list[tuple[str | Callable, Callable]] | None = None,
        reset_hooks_end: bool = True,
        clear_contexts: bool = False,
    ) -> Iterator["MaskedRunner"]:
        """Imitates the 'hooks' context manager in HookedTransformer."""
        with self.masked_transformer.hooks(
            fwd_hooks=fwd_hooks,
            bwd_hooks=bwd_hooks,
            reset_hooks_end=reset_hooks_end,
            clear_contexts=clear_contexts,
        ):
            yield self

    def run(
        self,
        input: Num[torch.Tensor, "batch pos"],
        patch_input: Num[torch.Tensor, "batch pos"] | None,
        edges_to_ablate: list[Edge],
    ) -> Num[torch.Tensor, "batch pos vocab"]:
        """If 'patch_input' is None, do not recalculate the ablation cache.
        Instead, use the ablation cache that has already been calculated.
        This is necessary for the 'run_with_linear_combination' method, but
        can also be useful for performance reason."""
        with self.with_ablated_edges(patch_input=patch_input, edges_to_ablate=edges_to_ablate) as hooked_model:
            return hooked_model(input)

    def run_with_linear_combination(
        self,
        input_embedded: Float[torch.Tensor, "batch pos d_resid"],
        dummy_input: Integer[torch.Tensor, " pos"],
        coefficients: Num[torch.Tensor, " batch"],
        patch_input: Integer[torch.Tensor, " pos"],
        edges_to_ablate: list[Edge],
    ) -> Float[torch.Tensor, "1 pos vocab"]:
        """'input_embedded' should be the input after the embedding layer.

        'dummy_input' and 'patch_input' should be a single input point (pre-embedding, without batch dim)"""

        def replace_embedding_with_convex_combination_hook(
            hook_point_out: torch.Tensor, hook: HookPoint, verbose=False
        ) -> Float[torch.Tensor, "1 pos d_resid"]:
            convex_combination = torch.einsum(
                "b, b p d -> p d",
                [torch.nn.functional.softmax(coefficients), input_embedded],
            ).unsqueeze(dim=0)
            return convex_combination

        self.masked_transformer.calculate_and_store_ablation_cache(patch_input, retain_cache_gradients=False)
        with self.hooks(
            fwd_hooks=[("hook_embed", replace_embedding_with_convex_combination_hook)]
        ) as runner_with_convex_combination:
            return runner_with_convex_combination.run(
                input=dummy_input.unsqueeze(0), patch_input=None, edges_to_ablate=edges_to_ablate
            )

    def run_with_linear_combination_of_input_and_patch(
        self,
        input_embedded: Float[torch.Tensor, "batch pos d_resid"],
        patch_input_embedded: Float[torch.Tensor, "batch pos d_resid"],
        dummy_input: Integer[torch.Tensor, " pos"],
        coefficients: Num[torch.Tensor, " batch"],
        coefficients_patch: Num[torch.Tensor, " batch"],
        edges_to_ablate: list[Edge],
        retain_patch_gradient: bool = False,
    ) -> Float[torch.Tensor, "1 pos vocab"]:
        """'input_embedded' should be the input after the embedding layer.

        'dummy_input' and 'patch_input' should be a single input point (pre-embedding, without batch dim)

        If 'retain_patch_gradient' is True, make sure to retain the gradient for the patch input by retaining the
        gradient in the ablation cache.
        """

        def replace_embedding_with_convex_combination_hook(
            hook_point_out: torch.Tensor, hook: HookPoint, verbose=False
        ) -> Float[torch.Tensor, "1 pos d_resid"]:
            convex_combination = torch.einsum(
                "b, b p d -> p d",
                [torch.nn.functional.softmax(coefficients), input_embedded],
            ).unsqueeze(dim=0)
            return convex_combination

        # todo cleanup: this is basically the same as the function above
        def replace_embedding_with_convex_combination_hook_patch(
            hook_point_out: torch.Tensor, hook: HookPoint, verbose=False
        ) -> Float[torch.Tensor, "1 pos d_resid"]:
            convex_combination = torch.einsum(
                "b, b p d -> p d",
                [torch.nn.functional.softmax(coefficients_patch), patch_input_embedded],
            ).unsqueeze(dim=0)
            return convex_combination

        # calculate ablation cache with the convex combination
        with self.hooks(
            fwd_hooks=[("hook_embed", replace_embedding_with_convex_combination_hook_patch)]
        ) as runner_with_convex_combination:
            runner_with_convex_combination.masked_transformer.calculate_and_store_ablation_cache(
                dummy_input.unsqueeze(0), retain_cache_gradients=retain_patch_gradient
            )

        # do forward pass with ablations on convex combination
        with self.hooks(
            fwd_hooks=[("hook_embed", replace_embedding_with_convex_combination_hook)]
        ) as runner_with_convex_combination:
            return runner_with_convex_combination.run(
                input=dummy_input.unsqueeze(0), patch_input=None, edges_to_ablate=edges_to_ablate
            )
