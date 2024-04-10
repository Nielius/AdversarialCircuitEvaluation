from abc import ABC
from dataclasses import dataclass
from enum import Enum
from functools import cached_property, partial
from typing import Callable

import torch
from jaxtyping import Float
from transformer_lens import HookedTransformer

from acdc.docstring.utils import AllDataThings, get_all_docstring_things, get_docstring_subgraph_true_edges
from acdc.greaterthan.utils import get_all_greaterthan_things, get_greaterthan_true_edges
from acdc.ioi.utils import get_all_ioi_things, get_ioi_true_edges
from acdc.nudb.adv_opt.loss_fn import kl_div_on_output_logits
from acdc.nudb.adv_opt.masked_runner import MaskedRunner
from acdc.nudb.adv_opt.utils import device
from acdc.TLACDCEdge import Edge
from acdc.tracr_task.utils import get_all_tracr_things, get_tracr_proportion_edges, get_tracr_reverse_edges
from acdc.types import EdgeAsTuple
from subnetwork_probing.masked_transformer import CircuitStartingPointType


def edge_tuples_to_dataclass(true_edges: dict[EdgeAsTuple, bool]) -> list[Edge]:
    return [Edge.from_tuple_format(*t) for t, is_present in true_edges.items() if is_present]


class AdvOptTaskName(str, Enum):
    TRACR_REVERSE = "tracr_reverse"
    TRACR_PROPORTION = "tracr_proportion"
    DOCSTRING = "docstring"
    INDUCTION = "induction"
    GREATERTHAN = "greaterthan"
    IOI = "ioi"


@dataclass
class AdvOptExperimentData:
    task_name: AdvOptTaskName
    task_data: AllDataThings
    circuit_edges: list[Edge]
    masked_runner: MaskedRunner
    loss_fn: Callable[
        [Float[torch.Tensor, "batch pos vocab"], Float[torch.Tensor, "batch pos vocab"]], Float[torch.Tensor, " batch"]
    ]

    @cached_property
    def ablated_edges(self) -> set[Edge]:
        # see comment at the bottom of this file about why I believe this works.
        return self.masked_runner.all_ablatable_edges - set(self.circuit_edges)

    @property
    def tokenizer(self):
        tokenizer = self.masked_runner.masked_transformer.model.tokenizer
        assert tokenizer is not None
        return tokenizer

    @property
    def embedder(self) -> torch.nn.Module:
        return self.masked_runner.masked_transformer.model.embed


class AdvOptDataProvider(ABC):
    def get_experiment_data(
        self,
        num_examples: int,
        metric_name: str,
        device: str,
    ) -> AdvOptExperimentData:
        raise NotImplementedError


def _determine_circuit_starting_point_type(circuit: list[Edge]) -> CircuitStartingPointType:
    """Returns the starting point type based on which hook point names occur in the true edges.

    Does a sanity check to see that it doesn't use both."""
    found_hook_pos: bool = False
    found_hook_resid_pre: bool = False

    for edge in circuit:
        if edge.parent.hook_name.endswith("hook_pos"):
            found_hook_pos = True

        if edge.parent.hook_name.endswith("hook_pos_embed"):
            found_hook_pos = True

        if edge.parent.hook_name.endswith("hook_resid_pre"):
            found_hook_resid_pre = True

    # We could do early stopping (return True/False as soon as we find one conclusive child), but just as a sanity
    # check we'll loop through all edges to exclude that both types occur:
    if found_hook_pos == found_hook_resid_pre:
        raise ValueError("Could not determine whether to use pos_embed or not.")

    return CircuitStartingPointType.POS_EMBED if found_hook_pos else CircuitStartingPointType.RESID_PRE


@dataclass
class ACDCAdvOptDataProvider(AdvOptDataProvider):
    task_data_fetcher: Callable[..., AllDataThings]
    true_edges_fetcher: Callable[[HookedTransformer], dict]
    task_name: AdvOptTaskName
    metric_last_sequence_position_only: bool = False

    def get_experiment_data(
        self,
        num_examples: int,
        metric_name: str,
        device: str,
    ) -> AdvOptExperimentData:
        task_data = self.task_data_fetcher(
            num_examples=num_examples,
            metric_name=metric_name,
            device=device,
        )
        true_edges = edge_tuples_to_dataclass(self.true_edges_fetcher(task_data.tl_model))

        return AdvOptExperimentData(
            task_name=self.task_name,
            task_data=task_data,
            circuit_edges=true_edges,
            masked_runner=MaskedRunner(
                model=task_data.tl_model, starting_point_type=_determine_circuit_starting_point_type(true_edges)
            ),
            loss_fn=partial(
                kl_div_on_output_logits, last_sequence_position_only=self.metric_last_sequence_position_only
            ),
        )


EXPERIMENT_DATA_PROVIDERS: dict[AdvOptTaskName, AdvOptDataProvider] = {
    AdvOptTaskName.TRACR_REVERSE: ACDCAdvOptDataProvider(
        # tracr-reverse is a task that takes a permutation of [0, 1, 2] and returns the reverse of it.
        # There are only 6 data points.
        # Only 1 attention head.
        # KL divergence is not well-defined for its output.
        task_name=AdvOptTaskName.TRACR_REVERSE,
        task_data_fetcher=lambda num_examples, metric_name, device: get_all_tracr_things(
            task="reverse", metric_name=metric_name, num_examples=num_examples, device=device
        ),
        true_edges_fetcher=lambda model: get_tracr_reverse_edges(),
    ),
    AdvOptTaskName.TRACR_PROPORTION: ACDCAdvOptDataProvider(
        # tracr-proportion is a task that takes a permutation of [0, 1, 2] and returns the proportion of the
        # first element in the permutation.
        task_name=AdvOptTaskName.TRACR_PROPORTION,
        task_data_fetcher=lambda num_examples, metric_name, device: get_all_tracr_things(
            task="proportion", metric_name=metric_name, num_examples=num_examples, device=device
        ),
        true_edges_fetcher=lambda model: get_tracr_proportion_edges(),
    ),
    AdvOptTaskName.DOCSTRING: ACDCAdvOptDataProvider(
        task_name=AdvOptTaskName.DOCSTRING,
        task_data_fetcher=lambda num_examples, metric_name, device: get_all_docstring_things(
            num_examples=num_examples, metric_name=metric_name, seq_len=4, device=device
        ),
        true_edges_fetcher=lambda model: get_docstring_subgraph_true_edges(),
        metric_last_sequence_position_only=True,
    ),
    AdvOptTaskName.GREATERTHAN: ACDCAdvOptDataProvider(
        task_name=AdvOptTaskName.GREATERTHAN,
        task_data_fetcher=lambda num_examples, metric_name, device: get_all_greaterthan_things(
            num_examples=num_examples, metric_name=metric_name, device=device
        ),
        true_edges_fetcher=get_greaterthan_true_edges,
        metric_last_sequence_position_only=True,
    ),
    AdvOptTaskName.IOI: ACDCAdvOptDataProvider(
        task_name=AdvOptTaskName.IOI,
        task_data_fetcher=lambda num_examples, metric_name, device: get_all_ioi_things(
            num_examples=num_examples, metric_name=metric_name, device=device
        ),
        true_edges_fetcher=get_ioi_true_edges,
        metric_last_sequence_position_only=True,
    ),
    # No canonical circuit for induction?
    # AdvOptExperimentName.INDUCTION
}

# Things you need to be careful about when translating an ACDC circuit to a set of edges you need to ablate.
#
# examples:
#
# 1. Make sure to check that the MaskedRunner's use_pos_embed is set correctly. Depending on its setting,
#    you get masks for either hook_embed & hook_pos_embed, or only for hook_resid_pre.
#    The ACDC's circuit will also only use one of those (I believe).
#
# 2. This is a list of edges that the ACDC correspondence includes, but that can't be masked. You don't
#    actually need to mask them necessarily.
#
#    attn-internal edges  (k/q/v-input -> k/q/v, and k/q/v -> result)
#       Edge(child=IndexedHookID(blocks.0.attn.hook_result, [:, :, 0]), parent=IndexedHookID(blocks.0.attn.hook_k, [:, :, 0]))
#                    (this is an edge of type "placeholder")                                                  also q,v
#       Edge(child=IndexedHookID(blocks.0.attn.hook_v, [:, :, 0]), parent=IndexedHookID(blocks.0.hook_v_input, [:, :, 0])),
#
#    mlp-internal (mlp_in -> mlp_out)
#       Edge(child=IndexedHookID(blocks.0.hook_mlp_out, [:]), parent=IndexedHookID(blocks.0.hook_mlp_in, [:])),
#
# 3. The statement "you don't need to mask them necessarily" in point 2. I think depends on whether or not
#    the ACDC circuit is left with dangling nodes (that are not on any path from input to output)
#    Any edge or not that is not included in at least one path from input to output can be safely ablated,
#    because ablating it has the same effect on the output as not ablating it.
#    I'm not 100% sure that the ACDC correspondence does not include such dangling edges/nodes, but I think so.


def get_standard_experiment_data(task_name: AdvOptTaskName) -> AdvOptExperimentData:
    experiment_data = EXPERIMENT_DATA_PROVIDERS[task_name].get_experiment_data(
        num_examples=1000 if task_name != AdvOptTaskName.TRACR_REVERSE else 30,
        metric_name="kl_div" if task_name != AdvOptTaskName.TRACR_REVERSE else "l2",
        device=device,
    )
    return experiment_data
