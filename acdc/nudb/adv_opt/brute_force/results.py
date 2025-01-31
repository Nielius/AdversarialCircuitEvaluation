import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from jaxtyping import Float

from acdc.ioi.ioi_dataset_v2 import IOI_PROMPT_PRETEMPLATES, IOI_PROMPT_PRETEMPLATES_OOD
from acdc.nudb.adv_opt.brute_force.circuit_edge_fetcher import CircuitSpec, CircuitType
from acdc.nudb.adv_opt.data_fetchers import AdvOptTaskName
from acdc.nudb.adv_opt.edge_serdes import EdgeJSONDecoder, EdgeJSONEncoder
from acdc.nudb.adv_opt.utils import device
from acdc.TLACDCEdge import Edge


@dataclass
class BruteForceResults:
    task_name: AdvOptTaskName
    input: Float[np.ndarray, "batch pos vocab"]
    patch_input: Float[np.ndarray, "batch pos vocab"]
    circuit_loss: Float[np.ndarray, " batch"]
    circuit_spec: CircuitSpec


@dataclass
class IOIBruteForceResults(BruteForceResults):
    prompt_template_index: int
    prompt_template: str
    task_name = AdvOptTaskName.IOI


@dataclass
class CircuitPerformanceDistributionResultsV1:
    """NOW DEPRECATED in favour of BruteForceResults."""

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
    ) -> "CircuitPerformanceDistributionResultsV1":
        storage_dir = artifact_dir / experiment_name if append_exp_name_to_dir else artifact_dir

        return cls(
            experiment_name=experiment_name,
            metrics={
                # load all metrics with torch.load by walking through the file names
                key: torch.load(storage_dir / f"metrics_{key}.pt", map_location=device)
                * (
                    50_257
                    if experiment_name in [AdvOptTaskName.IOI, AdvOptTaskName.DOCSTRING, AdvOptTaskName.GREATERTHAN]
                    else 1
                )
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

    def convert_to_brute_force_results(self) -> BruteForceResults:
        if self.experiment_name == AdvOptTaskName.IOI:
            # TECH DEBT: we need to find some way to recognize the correct template index from the input
            # Currently, I'm just doing a sanity check plus assuming it's the second index
            assert (
                self.test_data.shape[1] == 15
            ), "This does not seem to be template. Please fix the code here to make sure you can recognize the correct template index from the input."
            assert self.test_data[0, 5] == 1816  # " went"
            prompt_template_index = 2

            return IOIBruteForceResults(
                task_name=self.experiment_name,
                input=self.test_data.numpy(),
                patch_input=self.test_patch_data.numpy(),
                circuit_loss=self.metrics["canonical"].numpy(),
                circuit_spec=CircuitSpec(circuit_type=CircuitType.UNKNOWN, edges=self.random_circuit),
                prompt_template_index=prompt_template_index,
                prompt_template=(IOI_PROMPT_PRETEMPLATES + IOI_PROMPT_PRETEMPLATES_OOD)[prompt_template_index].template,
            )

        return BruteForceResults(
            task_name=self.experiment_name,
            input=self.test_data.numpy(),
            patch_input=self.test_patch_data.numpy(),
            circuit_loss=self.metrics["canonical"].numpy(),
            circuit_spec=CircuitSpec(circuit_type=CircuitType.UNKNOWN, edges=self.random_circuit),
        )
