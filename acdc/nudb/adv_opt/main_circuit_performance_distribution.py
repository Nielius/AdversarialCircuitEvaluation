import datetime
import itertools
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from jaxtyping import Float

from acdc.nudb.adv_opt.data_fetchers import AdvOptExperimentData, AdvOptTaskName, get_standard_experiment_data
from acdc.nudb.adv_opt.utils import CIRCUITBENCHMARKS_DATA_DIR, device
from acdc.TLACDCEdge import Edge

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.DEBUG)


@dataclass
class CircuitPerformanceDistributionExperiment:
    """A class to run a circuit performance distribution experiment.

    This means that we take a large sample of input points (sampled from test_data), and measure the circuit
    performance for each of them. The circuit performance of a circuit on an input point is defined
    by comparing the output of the circuit on the input point with the output of the full model on that input point."""

    experiment_data: AdvOptExperimentData

    def calculate_circuit_performance_for_large_sample(
        self,
        circuit: list[Edge],
    ) -> Float[torch.Tensor, " batch"]:
        """
        Run and calculate an individual circuit performance metrics for each input in `test_data`.
        This circuit performance metric compares the output for the circuit with the output for the full model, and it
        that way, it measures how well the circuit performs.

        'last_sequence_position_only' is a flag that should be set to True for tasks where only the last sequence position matters.
        If set to True, the metric will be calculated only for the last sequence position.
        Otherwise, the average metric will be calculated across all sequence positions.

        """
        masked_output_logits: Float[torch.Tensor, "batch pos vocab"] = self.experiment_data.masked_runner.run(
            input=self.experiment_data.task_data.test_data,
            patch_input=self.experiment_data.task_data.test_patch_data,
            edges_to_ablate=list(self.experiment_data.masked_runner.all_ablatable_edges - set(circuit)),
        )
        base_output_logits: Float[torch.Tensor, "batch pos vocab"] = self.experiment_data.masked_runner.run(
            input=self.experiment_data.task_data.test_data,
            patch_input=self.experiment_data.task_data.test_patch_data,
            edges_to_ablate=[],
        )

        return self.experiment_data.loss_fn(
            base_output_logits,
            masked_output_logits,
        )

    def random_circuit(self) -> list[Edge]:
        """TODO: this can be made smarter; e.g., we probably don't want to leave dangling nodes."""
        return [edge for edge in self.experiment_data.masked_runner.all_ablatable_edges if random.random() < 0.4]

    def canonical_circuit_with_random_edges_removed(self, num_of_removals: int) -> list[Edge]:
        return list(
            set(self.experiment_data.circuit_edges)
            - set(random.choices(self.experiment_data.circuit_edges, k=num_of_removals))
        )


@dataclass
class CircuitPerformanceDistributionResults:
    experiment_name: AdvOptTaskName
    metrics: dict[str, Float[torch.Tensor, " batch"]]
    topk_most_adversarial_values: list[float]
    topk_most_adversarial_input: list[str]

    def save(self, artifact_dir: Path):
        storage_dir = artifact_dir / self.experiment_name
        storage_dir.mkdir()
        for key, value in self.metrics.items():
            torch.save(value, storage_dir / f"metrics_{key}.pt")
        (storage_dir / "topk_most_adversarial.json").write_text(
            json.dumps(
                {
                    "topk_most_adversarial_values": self.topk_most_adversarial_values,
                    "topk_most_adversarial_input": self.topk_most_adversarial_input,
                }
            )
        )

    @classmethod
    def load(cls, artifact_dir: Path, experiment_name: AdvOptTaskName) -> "CircuitPerformanceDistributionResults":
        storage_dir = artifact_dir / experiment_name

        return cls(
            experiment_name=experiment_name,
            metrics={
                # load all metrics with torch.load by walking through the file names
                key: torch.load(storage_dir / f"metrics_{key}.pt")
                for filename in storage_dir.glob("metrics_*.pt")
                if (key := filename.removesuffix(".pt").removeprefix("metrics_"))
            },  # torch.load(storage_dir / "metrics.pt"),
            **json.loads((storage_dir / "topk_most_adversarial.json").read_text()),
        )

    def print(self):
        print(f"Experiment: {self.experiment_name}")
        print(f"Metrics: {self.metrics}")
        print(f"Topk most adversarial values: {self.topk_most_adversarial_values}")
        print(f"Topk most adversarial input: {self.topk_most_adversarial_input}")


def main_for_plotting_three_experiments(
    experiment_name: AdvOptTaskName,
    artifact_dir: Path,
    include_full_circuit: bool = False,  # this is only intended as a sanity check
) -> CircuitPerformanceDistributionResults:
    logger.info("Starting plotting experiment for '%s'.", experiment_name)
    experiment = CircuitPerformanceDistributionExperiment(experiment_data=get_standard_experiment_data(experiment_name))

    if include_full_circuit:
        logger.info("Running with all edges")
        metrics_with_full_model = experiment.calculate_circuit_performance_for_large_sample(
            circuit=list(experiment.experiment_data.masked_runner.all_ablatable_edges)
        ).to("cpu")
        # regarding '.to("cpu")': torch.histogram does not work for CUDA, so moving to CPU
        # see https://github.com/pytorch/pytorch/issues/69519

    logger.info("Running with canonical circuit")
    metrics_with_canonical_circuit = experiment.calculate_circuit_performance_for_large_sample(
        circuit=experiment.experiment_data.circuit_edges
    ).to("cpu")

    logger.info("Running with a random circuit")
    metrics_with_random_circuit = experiment.calculate_circuit_performance_for_large_sample(
        circuit=experiment.random_circuit()
    ).to("cpu")

    logger.info("Running with the canonical circuit, but with 2 random edges removed")
    metrics_with_corrupted_canonical_circuit = experiment.calculate_circuit_performance_for_large_sample(
        circuit=experiment.canonical_circuit_with_random_edges_removed(2)
    ).to("cpu")

    def plot():
        # plot histogram of output
        fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)  # Create a figure containing a single axes.
        ((ax_all, ax_1), (ax_2, ax_3)) = axes
        range = (
            0,
            max(
                # metrics_with_full_model.max().item(),  # can safely exclude this, as it's always supposed to be 0
                metrics_with_canonical_circuit.max().item(),
                metrics_with_random_circuit.max().item(),
                metrics_with_corrupted_canonical_circuit.max().item(),
            ),
        )
        if include_full_circuit:
            ax_all.stairs(*torch.histogram(metrics_with_full_model, bins=100, range=range), label="full model")
        ax_all.stairs(
            *torch.histogram(metrics_with_canonical_circuit, bins=100, range=range), label="canonical circuit"
        )
        ax_all.stairs(*torch.histogram(metrics_with_random_circuit, bins=100, range=range), label="random circuit")
        ax_all.stairs(
            *torch.histogram(metrics_with_corrupted_canonical_circuit, bins=100, range=range),
            label="corrupted canonical circuit",
        )
        fig.suptitle(
            f"KL divergence between output of the full model and output of a circuit, for {experiment_name}, histogram"
        )
        ax_1.stairs(*torch.histogram(metrics_with_canonical_circuit, bins=100, range=range), label="canonical circuit")
        ax_2.stairs(*torch.histogram(metrics_with_random_circuit, bins=100, range=range), label="random circuit")
        ax_3.stairs(
            *torch.histogram(metrics_with_corrupted_canonical_circuit, bins=100, range=range),
            label="corrupted canonical circuit",
        )
        for ax in itertools.chain(*axes):
            ax.set_xlabel("KL divergence")
            ax.set_ylabel("Frequency")
            ax.legend()

        plot_dir = artifact_dir / "plots"
        plot_dir.mkdir(exist_ok=True)
        figure_path = plot_dir / f"{experiment_name}_histogram_{datetime.datetime.now().isoformat()}.png"
        fig.savefig(figure_path)
        logger.info("Saved histogram to %s", figure_path)

    plot()

    topk_most_adversarial = torch.topk(metrics_with_random_circuit, k=5, sorted=True)
    topk_most_adversarial_input = experiment.experiment_data.task_data.test_data[topk_most_adversarial.indices, :]

    if experiment.experiment_data.masked_runner.masked_transformer.model.tokenizer is not None:
        # decode if a tokenizer is given
        topk_most_adversarial_input = [
            experiment.experiment_data.masked_runner.masked_transformer.model.tokenizer.decode(input)
            for input in topk_most_adversarial_input
        ]
    else:
        topk_most_adversarial_input = topk_most_adversarial_input.tolist()

    results = CircuitPerformanceDistributionResults(
        experiment_name=experiment_name,
        metrics={
            "random": metrics_with_random_circuit,
        },
        topk_most_adversarial_values=topk_most_adversarial.values.tolist(),
        topk_most_adversarial_input=topk_most_adversarial_input,
    )

    results.save(artifact_dir)
    results.print()

    # Debugging code: decode the input data with tokenizer
    # experiment.experiment_data.masked_runner.masked_transformer.model.tokenizer.decode(
    #     experiment.experiment_data.task_data.test_data[0]
    # )

    return results


if __name__ == "__main__":
    logger.info("Using device %s", device)

    artifact_dir = CIRCUITBENCHMARKS_DATA_DIR / f"run_{datetime.datetime.now().isoformat().replace(':', '-')}"
    artifact_dir.mkdir(exist_ok=True)

    # main_for_tracr_proportion()
    main_for_plotting_three_experiments(AdvOptTaskName.TRACR_REVERSE, artifact_dir)
    main_for_plotting_three_experiments(AdvOptTaskName.DOCSTRING, artifact_dir)
    main_for_plotting_three_experiments(AdvOptTaskName.GREATERTHAN, artifact_dir)
    main_for_plotting_three_experiments(AdvOptTaskName.IOI, artifact_dir)
