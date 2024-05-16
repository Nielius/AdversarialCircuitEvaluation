import logging
import random
import typing
from dataclasses import dataclass, field
from pathlib import Path

import hydra
import jsonpickle
import omegaconf
import torch
import torch.utils.data
from hydra.core import hydra_config
from hydra.core.config_store import ConfigStore
from jaxtyping import Float
from tqdm import tqdm

from acdc.ioi.ioi_dataset_v2 import IOI_PROMPT_PRETEMPLATES
from acdc.nudb.adv_opt.brute_force.circuit_edge_fetcher import CircuitType, get_circuit_edges
from acdc.nudb.adv_opt.brute_force.results import (
    BruteForceResults,
    IOIBruteForceResults,
)
from acdc.nudb.adv_opt.data_fetchers import (
    AdvOptExperimentData,
    AdvOptTaskName,
    get_adv_opt_data_provider_for_ioi,
    get_standard_experiment_data,
)
from acdc.nudb.adv_opt.masked_runner import MaskedRunner
from acdc.nudb.adv_opt.settings import IOITaskSpecificSettings, TaskSpecificSettings, TracrReverseTaskSpecificSettings
from acdc.nudb.adv_opt.utils import device
from acdc.TLACDCEdge import Edge

logger = logging.getLogger(__name__)


@dataclass
class BruteForceExperimentSettings:
    """Settings for the brute force experiment."""

    task: TaskSpecificSettings
    seed: int = 4321
    batch_size: int = 256
    optimize_over_patch_data: bool = True
    circuits: list[CircuitType] = field(
        default_factory=lambda: [CircuitType.RANDOM, CircuitType.CANONICAL, CircuitType.CORRUPTED_CANONICAL]
    )
    num_examples: int | None = None


cs = ConfigStore.instance()
cs.store(name="config_schema", node=BruteForceExperimentSettings)
# Not sure if/why you need to add the two lines below. Maybe just for checking that config schema?
# (I'm now 80% sure that these things are only for schema matching.)
cs.store(group="task", name="greaterthan_schema", node=TaskSpecificSettings)
cs.store(group="task", name="tracr_reverse_schema", node=TracrReverseTaskSpecificSettings)
cs.store(group="task", name="ioi_schema", node=IOITaskSpecificSettings)


@dataclass
class CircuitPerformanceDistributionExperiment:
    """A class to run a circuit performance distribution experiment.

    This means that we take a large sample of input points (sampled from test_data), and measure the circuit
    performance for each of them. The circuit performance of a circuit on an input point is defined
    by comparing the output of the circuit on the input point with the output of the full model on that input point."""

    masked_runner: MaskedRunner
    loss_fn: typing.Callable[
        [Float[torch.Tensor, "batch pos vocab"], Float[torch.Tensor, "batch pos vocab"]], Float[torch.Tensor, " batch"]
    ]

    @classmethod
    def from_experiment_data(cls, experiment_data: AdvOptExperimentData):
        return cls(masked_runner=experiment_data.masked_runner, loss_fn=experiment_data.loss_fn)

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

            masked_output_logits: Float[torch.Tensor, "batch pos vocab"] = self.masked_runner.run(
                input=input,
                patch_input=patch_input,
                edges_to_ablate=list(self.masked_runner.all_ablatable_edges - set(circuit)),
            )
            base_output_logits: Float[torch.Tensor, "batch pos vocab"] = self.masked_runner.run(
                input=input,
                patch_input=patch_input,
                edges_to_ablate=[],
            )

            return self.loss_fn(
                base_output_logits,
                masked_output_logits,
            )

        return torch.cat([process_batch(batch) for batch in tqdm(data_loader)])


def create_cartesian_product(
    test_data: torch.Tensor, test_patch_data: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.repeat_interleave(test_data, len(test_patch_data), dim=0),
        torch.cat(len(test_data) * [test_patch_data]),
    )


@hydra.main(config_path="../conf", config_name="config_bruteforce", version_base=None)
def main(
    settings: BruteForceExperimentSettings,
) -> None:
    settings_object = typing.cast(BruteForceExperimentSettings, omegaconf.OmegaConf.to_object(settings))
    experiment_name = settings.task.task_name
    output_base_dir = Path(hydra_config.HydraConfig.get().runtime.output_dir)
    logger.info("Starting brute force experiment for '%s'.", experiment_name)
    logger.info("Storing output in %s; going to run circuits %s", output_base_dir, settings.circuits)

    rng = random.Random(settings.seed)

    if experiment_name == AdvOptTaskName.IOI:
        assert isinstance(settings_object.task, IOITaskSpecificSettings)
        experiment_data = get_standard_experiment_data(
            task_name=experiment_name,
            data_provider=get_adv_opt_data_provider_for_ioi(
                rng=rng,
                template_index=settings_object.task.prompt_template_index,
                names_order=settings_object.task.names_order,
            ),
            num_examples=settings_object.num_examples,
        )
    else:
        experiment_data = get_standard_experiment_data(experiment_name)
    experiment = CircuitPerformanceDistributionExperiment.from_experiment_data(experiment_data)

    test_data, test_patch_data = (
        (experiment_data.task_data.test_data, experiment_data.task_data.test_patch_data)
        if not settings.optimize_over_patch_data
        else create_cartesian_product(experiment_data.task_data.test_data, experiment_data.task_data.test_patch_data)
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

    for circuit in settings.circuits:
        logger.info(f"Running with circuit {circuit}")
        circuit_spec = get_circuit_edges(
            rng=random.Random(settings.seed),
            # use a new rng here; makes it easier to replicate; e.g. independent of iteration order, and independent of data selection
            experiment_data=experiment_data,
            circuit_type=circuit,
        )
        metrics_for_circuit = experiment.calculate_circuit_performance_for_large_sample(
            circuit=circuit_spec.edges,
            data_loader=data_loader,
        ).to("cpu")
        # regarding '.to("cpu")': torch.histogram does not work for CUDA, so moving to CPU
        # see https://github.com/pytorch/pytorch/issues/69519

        if experiment_name == AdvOptTaskName.IOI:
            assert isinstance(settings_object.task, IOITaskSpecificSettings)
            results = IOIBruteForceResults(
                task_name=experiment_name,
                input=test_data.detach().cpu().numpy(),
                patch_input=test_patch_data.detach().cpu().numpy(),
                circuit_spec=circuit_spec,
                circuit_loss=metrics_for_circuit.detach().cpu().numpy(),
                prompt_template_index=settings_object.task.prompt_template_index,
                prompt_template=IOI_PROMPT_PRETEMPLATES[settings_object.task.prompt_template_index].template,
            )
        else:
            results = BruteForceResults(
                task_name=experiment_name,
                input=test_data.detach().cpu().numpy(),
                patch_input=test_patch_data.detach().cpu().numpy(),
                circuit_spec=circuit_spec,
                circuit_loss=metrics_for_circuit.detach().cpu().numpy(),
            )

        results_file = output_base_dir / f"results_{str(circuit).lower()}.json"
        results_file.write_text(jsonpickle.encode(results))
        logger.info("Results stored in %s", output_base_dir)


if __name__ == "__main__":
    logger.info("Using device %s", device)

    main()
