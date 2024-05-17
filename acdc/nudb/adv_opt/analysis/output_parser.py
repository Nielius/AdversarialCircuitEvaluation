import re
from functools import cached_property
from pathlib import Path
from typing import ClassVar

import jsonpickle
import torch
import yaml

from acdc.nudb.adv_opt.brute_force.circuit_edge_fetcher import CircuitType
from acdc.nudb.adv_opt.brute_force.results import BruteForceResults
from acdc.nudb.adv_opt.data_fetchers import AdvOptExperimentData, AdvOptTaskName, get_standard_experiment_data
from acdc.nudb.adv_opt.settings import ExperimentArtifacts


class HydraConfig:
    """Represents the config.yaml file that Hydra writes to disk."""

    cfg_dict: dict

    def __init__(self, cfg_dict: dict):
        self.cfg_dict = cfg_dict

    @cached_property
    def task_name(self) -> AdvOptTaskName:
        return AdvOptTaskName[self.cfg_dict["task"]["task_name"]]

    @classmethod
    def from_file(cls, path: Path) -> "HydraConfig":
        return cls(yaml.safe_load(path.read_text()))


class HydraOutputDir:
    path: Path

    def __init__(self, path: Path):
        self.path = path

    @cached_property
    def config(self) -> HydraConfig:
        return HydraConfig.from_file(self.path / ".hydra" / "config.yaml")


class AdvOptHydraOutputDir(HydraOutputDir):
    """This class represents an output directory for the optimization (not the brute force)
    process, i.e., where I try to find the worst case input using an optimization process.
    These output directories are created by the file 'acdc/nudb/adv_opt/main.py'."""

    @cached_property
    def artifacts(self) -> ExperimentArtifacts:
        return ExperimentArtifacts.load(self.path / "artifacts")

    @cached_property
    def standard_experiment_data_for_task(self) -> AdvOptExperimentData:
        return get_standard_experiment_data(self.config.task_name)

    def analyze(self):
        """Note: this code is basically a scratch pad."""
        coeffs = self.artifacts.coefficients_final
        experiment_data = self.standard_experiment_data_for_task
        assert coeffs is not None
        base_input = self.artifacts.base_input
        assert base_input is not None
        topk = torch.topk(coeffs, 10)
        tokenizer = experiment_data.masked_runner.masked_transformer.model.tokenizer
        assert tokenizer is not None
        for token in base_input[topk.indices, :]:
            print(tokenizer.decode(token))

        patch_input = experiment_data.task_data.test_patch_data[0, ...]

        outputs_from_circuit = experiment_data.masked_runner.run(
            input=base_input[topk.indices, :],
            patch_input=patch_input,
            edges_to_ablate=list(experiment_data.ablated_edges),
        )
        outputs_from_full_model = experiment_data.masked_runner.run(
            input=base_input[topk.indices, :],
            patch_input=patch_input,
            edges_to_ablate=[],
        )

        loss = experiment_data.loss_fn(outputs_from_circuit, outputs_from_full_model)

        return loss


class AdvOptBruteForceOutputDir(HydraOutputDir):
    """This class represents an output directory for the brute force process, i.e., where I evaluate the performance of
    a circuit on a dataset. These output directories are created by the file 'acdc/nudb/adv_opt/brute_force/main.py'."""

    _results_re: ClassVar = re.compile(r"results_circuittype\.(.*)\.json")

    @property
    def result_paths(self) -> dict[CircuitType, Path]:
        return {self._circuit_type_from_path(path): path for path in self.path.glob("results_*.json")}

    def result(self, circuit_type: CircuitType) -> BruteForceResults:
        try:
            return jsonpickle.decode(self.result_paths[circuit_type].read_text())
        except KeyError:
            raise ValueError(f"No results found for circuit type {circuit_type} in {self.path}")

    @staticmethod
    def _circuit_type_from_path(path: Path) -> CircuitType:
        return CircuitType(AdvOptBruteForceOutputDir._results_re.match(path.name).group(1))
