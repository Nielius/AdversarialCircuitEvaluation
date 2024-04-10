import json
import logging
import random
import typing
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import hydra
import torch
import torch.utils.data
from hydra.core import hydra_config
from hydra.core.config_store import ConfigStore
from jaxtyping import Float
from tqdm import tqdm

from acdc.nudb.adv_opt.data_fetchers import AdvOptExperimentData, AdvOptTaskName, get_standard_experiment_data
from acdc.nudb.adv_opt.edge_serdes import EdgeJSONDecoder, EdgeJSONEncoder
from acdc.nudb.adv_opt.settings import TaskSpecificSettings, TracrReverseTaskSpecificSettings
from acdc.nudb.adv_opt.utils import device
from acdc.TLACDCEdge import Edge

logger = logging.getLogger(__name__)


class CirctuitType(str, Enum):
    RANDOM = "random"
    CANONICAL = "canonical"
    CORRUPTED_CANONICAL = "corrupted_canonical"
    FULL_MODEL = "full_model"


@dataclass
class BruteForceExperimentSettings:
    """Settings for the brute force experiment."""

    task: TaskSpecificSettings
    batch_size: int = 256
    optimize_over_patch_data: bool = True
    circuits: list[CirctuitType] = field(
        default_factory=lambda: [CirctuitType.RANDOM, CirctuitType.CANONICAL, CirctuitType.CORRUPTED_CANONICAL]
    )


cs = ConfigStore.instance()
cs.store(name="config_schema", node=BruteForceExperimentSettings)
# Not sure if/why you need to add the two lines below. Maybe just for checking that config schema?
# (I'm now 80% sure that these things are only for schema matching.)
cs.store(group="task", name="greaterthan_schema", node=TaskSpecificSettings)
cs.store(group="task", name="tracr_reverse_schema", node=TracrReverseTaskSpecificSettings)


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
        data_loader: torch.utils.data.DataLoader[
            tuple[
                Float[torch.Tensor, "batch pos vocab"],
                Float[torch.Tensor, "batch pos vocab"],
            ]
        ],  # tuples of (input, patch_input)
    ) -> Float[torch.Tensor, " batch"]:
        """
        Run and calculate an individual circuit performance metrics for each input in `test_data`.
        This circuit performance metric compares the output for the circuit with the output for the full model, and it
        that way, it measures how well the circuit performs.

        'last_sequence_position_only' is a flag that should be set to True for tasks where only the last sequence position matters.
        If set to True, the metric will be calculated only for the last sequence position.
        Otherwise, the average metric will be calculated across all sequence positions.

        """

        def process_batch(
            batch: tuple[Float[torch.Tensor, "batch pos vocab"], Float[torch.Tensor, "batch pos vocab"]],
        ) -> Float[torch.Tensor, " batch"]:
            input, patch_input = batch

            masked_output_logits: Float[torch.Tensor, "batch pos vocab"] = self.experiment_data.masked_runner.run(
                input=input,
                patch_input=patch_input,
                edges_to_ablate=list(self.experiment_data.masked_runner.all_ablatable_edges - set(circuit)),
            )
            base_output_logits: Float[torch.Tensor, "batch pos vocab"] = self.experiment_data.masked_runner.run(
                input=input,
                patch_input=patch_input,
                edges_to_ablate=[],
            )

            return self.experiment_data.loss_fn(
                base_output_logits,
                masked_output_logits,
            )

        return torch.cat([process_batch(batch) for batch in tqdm(data_loader)])

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
    test_data: Float[torch.Tensor, "batch pos vocab"]
    test_patch_data: Float[torch.Tensor, "batch pos vocab"]
    random_circuit: list[Edge]

    def save(self, artifact_dir: Path):
        artifact_dir.mkdir()
        for key, value in self.metrics.items():
            torch.save(value, artifact_dir / f"metrics_{key}.pt")
        torch.save(self.test_data, artifact_dir / "test_data.pt")
        torch.save(self.test_patch_data, artifact_dir / "test_patch_data.pt")
        (artifact_dir / "random_circuit.json").write_text(json.dumps(self.random_circuit, cls=EdgeJSONEncoder))

    @classmethod
    def load(
        cls, artifact_dir: Path, experiment_name: AdvOptTaskName, append_exp_name_to_dir: bool = True
    ) -> "CircuitPerformanceDistributionResults":
        storage_dir = artifact_dir / experiment_name if append_exp_name_to_dir else artifact_dir

        return cls(
            experiment_name=experiment_name,
            metrics={
                # load all metrics with torch.load by walking through the file names
                key: torch.load(storage_dir / f"metrics_{key}.pt", map_location=device)
                for filename in storage_dir.glob("metrics_*.pt")
                if (key := filename.stem.removeprefix("metrics_"))
            },  # torch.load(storage_dir / "metrics.pt"),
            test_data=torch.load(storage_dir / "test_data.pt", map_location=device),
            test_patch_data=torch.load(storage_dir / "test_patch_data.pt", map_location=device),
            random_circuit=json.loads((storage_dir / "random_circuit.json").read_text(), cls=EdgeJSONDecoder),
        )

    def print(self):
        print(f"Experiment: {self.experiment_name}")
        print(f"Metrics: {self.metrics}")
        # print(f"Topk most adversarial values: {self.topk_most_adversarial_values}")
        # print(f"Topk most adversarial inputs: {self.topk_most_adversarial_input}")


@hydra.main(config_path="conf", config_name="config_bruteforce", version_base=None)
def main(
    settings: BruteForceExperimentSettings,
) -> CircuitPerformanceDistributionResults:
    experiment_name = settings.task.task_name
    output_base_dir = Path(hydra_config.HydraConfig.get().runtime.output_dir)
    artifact_dir = output_base_dir / "artifacts"
    logger.info("Starting plotting experiment for '%s'.", experiment_name)
    logger.info("Storing output in %s; going to run circuits %s", output_base_dir, settings.circuits)

    experiment = CircuitPerformanceDistributionExperiment(experiment_data=get_standard_experiment_data(experiment_name))

    def create_cartesian_product(
        test_data: torch.Tensor, test_patch_data: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.repeat_interleave(test_data, len(test_patch_data), dim=0),
            torch.cat(len(test_data) * [test_patch_data]),
        )

    test_data, test_patch_data = (
        (experiment.experiment_data.task_data.test_data, experiment.experiment_data.task_data.test_patch_data)
        if not settings.optimize_over_patch_data
        else create_cartesian_product(
            experiment.experiment_data.task_data.test_data, experiment.experiment_data.task_data.test_patch_data
        )
    )

    data_loader = typing.cast(
        torch.utils.data.DataLoader[
            tuple[Float[torch.Tensor, "batch pos vocab"], Float[torch.Tensor, "batch pos vocab"]]
        ],
        torch.utils.data.DataLoader(
            dataset=torch.utils.data.TensorDataset(test_data, test_patch_data),
            batch_size=settings.batch_size,
        ),
    )

    collected_metrics: dict[str, Float[torch.Tensor, " batch"]] = {}

    random_circuit = experiment.random_circuit()

    def get_circuit_edges(circuit_type: CirctuitType) -> list[Edge]:
        match circuit_type:
            case CirctuitType.RANDOM:
                return random_circuit
            case CirctuitType.CANONICAL:
                return experiment.experiment_data.circuit_edges
            case CirctuitType.FULL_MODEL:
                return list(experiment.experiment_data.masked_runner.all_ablatable_edges)
            case CirctuitType.CORRUPTED_CANONICAL:
                return experiment.canonical_circuit_with_random_edges_removed(2)
            case _:
                raise ValueError(f"Unknown circuit type: {circuit_type}")

    for circuit in settings.circuits:
        logger.info(f"Running with circuit {circuit}")
        metrics_for_circuit = experiment.calculate_circuit_performance_for_large_sample(
            circuit=get_circuit_edges(circuit),
            data_loader=data_loader,
        ).to("cpu")
        # regarding '.to("cpu")': torch.histogram does not work for CUDA, so moving to CPU
        # see https://github.com/pytorch/pytorch/issues/69519
        collected_metrics[circuit] = metrics_for_circuit

    results = CircuitPerformanceDistributionResults(
        experiment_name=experiment_name,
        metrics=collected_metrics,
        test_data=test_data,
        test_patch_data=test_patch_data,
        random_circuit=random_circuit,
    )

    results.save(artifact_dir)
    results.print()

    logger.info("Output stored in %s", output_base_dir)

    return results


if __name__ == "__main__":
    logger.info("Using device %s", device)

    main()
