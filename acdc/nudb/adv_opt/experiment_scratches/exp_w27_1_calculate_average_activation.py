# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import typing
from typing import Callable, Sequence

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.utils.data
from jaxtyping import Float, Num
from transformer_lens import ActivationCache, HookedTransformer

from acdc.nudb.adv_opt.brute_force.main import CircuitPerformanceDistributionExperiment
from acdc.nudb.adv_opt.data_fetchers import AdvOptTaskName, get_standard_experiment_data
from acdc.TLACDCEdge import Edge, IndexedHookPointName

# %%
task_name_str = "docstring"

# %%
task_name = AdvOptTaskName(task_name_str)

# %%


def extract_keyed_activations_from_cache(
    cache: ActivationCache, hook_points_to_monitor: Sequence[IndexedHookPointName]
) -> dict[IndexedHookPointName, Float[torch.Tensor, "batch pos vocab"]]:
    return {hook_point: cache[hook_point.hook_name][hook_point.index.as_index] for hook_point in hook_points_to_monitor}


def compare_keyed_activations(
    base: dict[IndexedHookPointName, Float[torch.Tensor, "pos vocab"]],
    other: dict[IndexedHookPointName, Float[torch.Tensor, "batch pos vocab"]],
) -> dict[IndexedHookPointName, Float[torch.Tensor, "batch pos"]]:
    return {hook_point: F.cosine_similarity(base[hook_point], other[hook_point], dim=2) for hook_point in base.keys()}


def calculate_cosine_similarity_measure(
    base: dict[IndexedHookPointName, Float[torch.Tensor, "pos vocab"]],
    other: dict[IndexedHookPointName, Float[torch.Tensor, "batch pos vocab"]],  # <- these are the indexed activations
) -> Float[torch.Tensor, " batch"]:
    keyed_comparison = compare_keyed_activations(base, other)

    # take the mean over the positions
    cosine_similarity_mean_by_hook = {
        hook_point: keyed_comparison[hook_point].mean(dim=-1) for hook_point in keyed_comparison.keys()
    }

    # now take the mean over all the hooks
    return torch.stack(list(cosine_similarity_mean_by_hook.values()), dim=0).mean(dim=0)


def calculate_l2_distance_measure(
    base: dict[IndexedHookPointName, Float[torch.Tensor, "pos vocab"]],
    other: dict[IndexedHookPointName, Float[torch.Tensor, "batch pos vocab"]],  # <- these are the indexed activations
) -> Float[torch.Tensor, " batch"]:
    l2_distance_by_hook = {
        hook_point: (base[hook_point] - other[hook_point]).norm(dim=2).norm(dim=1) for hook_point in base.keys()
    }

    # now sum over all the hooks
    return torch.stack(list(l2_distance_by_hook.values()), dim=0).sum(dim=0)


# Steps in the script:

# 1. Calculate all the similarity measures
# 2. Calculate the performance of each individual point
# 3. Plot the performance against the similarity measure

# I want to go through this entire flow at least once, using the patched input that is provided by the task data.


class CircuitActivationCalculator:
    tl_model: HookedTransformer

    _hook_points_to_monitor: Sequence[IndexedHookPointName]
    _names_filter: typing.Callable[[str], bool]

    def __init__(
        self,
        tl_model: HookedTransformer,
        circuit_edges: Sequence[Edge],
    ):
        self.tl_model = tl_model
        self._hook_points_to_monitor = [edge.parent for edge in circuit_edges]

        names_of_hook_points_to_monitor = set([hook_point.hook_name for hook_point in self._hook_points_to_monitor])

        def hook_point_name_filter(hook_name: str) -> bool:
            return hook_name in names_of_hook_points_to_monitor

        self._names_filter = hook_point_name_filter

    def calculate_activations_on_circuit(
        self,
        input: Num[torch.Tensor, "batch pos"],
    ) -> dict[IndexedHookPointName, Float[torch.Tensor, "pos vocab"]]:
        _, cache = self.tl_model.run_with_cache(input, return_cache_object=True, names_filter=self._names_filter)
        new_activations = extract_keyed_activations_from_cache(cache, self._hook_points_to_monitor)

        return new_activations

    def calculate_mean_activations_on_circuit(
        self,
        data_loader: torch.utils.data.DataLoader,
    ) -> dict[IndexedHookPointName, Float[torch.Tensor, "pos vocab"]]:
        num_inputs = 0
        summed_activations = {}

        for i, batch in enumerate(data_loader):
            num_inputs += batch[0].shape[0]

            new_activations = self.calculate_activations_on_circuit(batch[0])

            for hook_point, activation in new_activations.items():
                if hook_point not in summed_activations:
                    summed_activations[hook_point] = activation.sum(dim=0)
                else:
                    summed_activations[hook_point] += activation.sum(dim=0)

        mean_activations = {
            hook_point: summed_activation / num_inputs for hook_point, summed_activation in summed_activations.items()
        }

        return mean_activations


def calculate_similarity_measure_batched(
    base: dict[IndexedHookPointName, Float[torch.Tensor, "pos vocab"]],
    data_loader: torch.utils.data.DataLoader,
    circuit_activation_calculator: CircuitActivationCalculator,
    similarity_measure: Callable,  # see calculate_cosine_similarity_measure for the type
) -> Float[torch.Tensor, " batch"]:
    return torch.cat(
        [
            similarity_measure(base, circuit_activation_calculator.calculate_activations_on_circuit(input))
            for input, _ in data_loader
        ],
    )


# %%
experiment_data = get_standard_experiment_data(task_name)


# %%

batch_size = 128
data_loader = typing.cast(
    torch.utils.data.DataLoader[tuple[Float[torch.Tensor, "batch pos vocab"], Float[torch.Tensor, "batch pos vocab"]]],
    torch.utils.data.DataLoader(
        dataset=torch.utils.data.TensorDataset(
            experiment_data.task_data.test_data, experiment_data.task_data.test_patch_data
        ),
        batch_size=batch_size,
    ),
)

# %%

tl_model = experiment_data.masked_runner.masked_transformer.model  # the underlying model, without any hooks
circuit_edges = experiment_data.circuit_edges

mean_activations = CircuitActivationCalculator(tl_model, circuit_edges).calculate_mean_activations_on_circuit(
    data_loader
)


# %%

circuit_performance = CircuitPerformanceDistributionExperiment.from_experiment_data(
    experiment_data
).calculate_circuit_performance_for_large_sample(circuit=circuit_edges, data_loader=data_loader)

# %%

cosine_similarities = calculate_similarity_measure_batched(
    mean_activations,
    data_loader,
    CircuitActivationCalculator(tl_model, circuit_edges),
    similarity_measure=calculate_cosine_similarity_measure,
)

# %%

l2_distances = calculate_similarity_measure_batched(
    mean_activations,
    data_loader,
    CircuitActivationCalculator(tl_model, circuit_edges),
    similarity_measure=calculate_l2_distance_measure,
)

# %%
# Plot the performance against the cosine similarity

plt.scatter(cosine_similarities, circuit_performance)
# %%
plt.scatter(l2_distances, circuit_performance)

# Now, let's try to use some other similarity measure, like the l2 distance.
