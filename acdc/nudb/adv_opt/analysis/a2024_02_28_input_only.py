"""
Analyse data for the convex combination experiments, where I try to find the worst input for a circuit by relaxing
the problem by allowing convex combinations of inputs, and then doing gradient descent on the coefficients of the
convex combination.

In this file, I analyze the experiments where we only have convex combinations of the inputs, not of the patch inputs.
"""
import logging
from dataclasses import dataclass
from pathlib import Path

import hydra
import torch
import yaml
from hydra.core.config_store import ConfigStore
from icecream import ic
from jaxtyping import Float

from acdc.nudb.adv_opt.data_fetchers import AdvOptExperimentData, AdvOptTaskName, get_standard_experiment_data
from acdc.nudb.adv_opt.settings import ExperimentArtifacts
from acdc.nudb.adv_opt.utils import deep_map

base_dir = Path("/home/niels/proj/mats/data/outputs/input_only")

# useful experiments to look at
# All have the same Adam LR of 0.1, and 400 epochs, but different random seeds
experiment_paths: dict[AdvOptTaskName, list[str]] = {
    AdvOptTaskName.GREATERTHAN: [
        "2024-02-29-042942-greaterthan_seed_4329/4329",
        "2024-02-29-042942-greaterthan_seed_4323/4323",
        "2024-02-29-042942-greaterthan_seed_4328/4328",
        "2024-02-29-042942-greaterthan_seed_4326/4326",
        "2024-02-29-042942-greaterthan_seed_4327/4327",
        "2024-02-29-042942-greaterthan_seed_4330/4330",
        "2024-02-29-042942-greaterthan_seed_4321/4321",
        "2024-02-29-042942-greaterthan_seed_4324/4324",
        "2024-02-29-042942-greaterthan_seed_4325/4325",
        "2024-02-29-042942-greaterthan_seed_4322/4322",
    ],
    AdvOptTaskName.IOI: [
        "2024-02-28-051332-ioi_seed_4321/4321",
        "2024-02-28-051332-ioi_seed_4322/4322",
        "2024-02-28-051332-ioi_seed_4323/4323",
        "2024-02-28-051332-ioi_seed_4324/4324",
        "2024-02-28-051332-ioi_seed_4325/4325",
        "2024-02-28-051332-ioi_seed_4326/4326",
        "2024-02-28-051332-ioi_seed_4327/4327",
        "2024-02-28-051332-ioi_seed_4328/4328",
        "2024-02-28-051332-ioi_seed_4329/4329",
        "2024-02-28-051332-ioi_seed_4330/4330",
    ],
}


@dataclass
class ExperimentAnalysis:
    """Analyse the artifacts from ExperimentArtifacts. See its documentation."""

    artifacts: ExperimentArtifacts
    config: dict

    @classmethod
    def from_dir(cls, experiment_dir: Path) -> "ExperimentAnalysis":
        return cls(
            artifacts=ExperimentArtifacts.load(experiment_dir / "artifacts"),
            config=yaml.safe_load((experiment_dir / ".hydra" / "config.yaml").read_text()),
        )

    def get_experiment_data(self) -> AdvOptExperimentData:
        return get_standard_experiment_data(task_name=AdvOptTaskName[self.config["task"]["task_name"]])

    def get_embedder(self) -> torch.nn.Module:
        experiment_data = get_standard_experiment_data(task_name=AdvOptTaskName[self.config["task"]["task_name"]])
        return experiment_data.masked_runner.masked_transformer.model.embed

    def get_tokenizer(self):
        experiment_data = get_standard_experiment_data(task_name=AdvOptTaskName[self.config["task"]["task_name"]])
        return experiment_data.masked_runner.masked_transformer.model.tokenizer

    def topk_inputs(self, k: int = 10, final_coefficients: bool = True) -> torch.return_types.topk:
        """
        Return the base inputs with the top k highest coefficients among the final coefficients.

        Use the initial coefficients instead of the final coefficients if final_coefficients is False."""
        coeffs = self.artifacts.coefficients_final if final_coefficients else self.artifacts.coefficients_init
        assert coeffs is not None
        return torch.topk(coeffs, k)

    def topk_inputs_decoded(self, tokenizer, k: int = 10, final_coefficients: bool = True) -> list[str]:
        topk = self.topk_inputs(k, final_coefficients)
        assert self.artifacts.base_input is not None
        return [tokenizer.decode(token) for token in self.artifacts.base_input[topk.indices, :]]

    def topk_input_score(
        self, experiment_data: AdvOptExperimentData, k: int = 10, final_coefficients: bool = True
    ) -> Float[torch.Tensor, " batch"]:
        topk = self.topk_inputs(k, final_coefficients)
        coefficients = self.artifacts.coefficients_final
        assert coefficients is not None
        return coefficients[topk.indices]

    def outputs_for_topk_inputs(
        self, experiment_data: AdvOptExperimentData, k: int = 10, circuit: bool = True, final_coefficients: bool = True
    ) -> Float[torch.Tensor, "batch pos vocab"]:
        """Get output for full model if `circuit` is set to False."""
        topk = self.topk_inputs(k, final_coefficients)
        # Decode input:
        # [experiment_data.tokenizer.decode(token) for token in experiment_data.task_data.test_patch_data[0, ...]]
        assert self.artifacts.base_input is not None
        return experiment_data.masked_runner.run(
            input=self.artifacts.base_input[topk.indices, :],
            patch_input=experiment_data.task_data.test_patch_data[0, ...],
            edges_to_ablate=list(experiment_data.ablated_edges) if circuit else [],
        )

    def outputs_for_topk_inputs_decoded(
        self,
        experiment_data: AdvOptExperimentData,
        k_inputs: int = 10,
        k_outputs: int = 10,
        circuit: bool = True,
        final_coefficients: bool = True,
    ) -> Float[torch.Tensor, "batch pos vocab"]:
        """For each of the k_inputs worst inputs, get the k_outputs most likely output values, decoded."""
        outputs = self.outputs_for_topk_inputs(
            experiment_data, k_inputs, circuit, final_coefficients=final_coefficients
        )
        return deep_map(experiment_data.tokenizer.decode, torch.topk(outputs[:, -1, :], k=k_outputs).indices.tolist())

    def loss_for_topk_inputs(
        self,
        experiment_data: AdvOptExperimentData,
        k: int = 10,
    ) -> Float[torch.Tensor, " batch"]:
        outputs_circuit = self.outputs_for_topk_inputs(experiment_data, k, circuit=True)
        outputs_full_model = self.outputs_for_topk_inputs(experiment_data, k, circuit=False)
        return experiment_data.loss_fn(outputs_circuit, outputs_full_model)


@dataclass
class Settings:
    task_name: AdvOptTaskName = AdvOptTaskName.GREATERTHAN


cs = ConfigStore.instance()
cs.store(name="settings_schema", node=Settings)

logger = logging.getLogger(__name__)
ic.configureOutput(outputFunction=logger.info)


@hydra.main(config_name="2024_02_28_input_only")
def main(settings: Settings):
    task_name = settings.task_name
    experiment_analyses = [
        ExperimentAnalysis.from_dir(base_dir / experiment_dir) for experiment_dir in experiment_paths[task_name]
    ]
    experiment_data = experiment_analyses[0].get_experiment_data()

    for i, analysis in enumerate(experiment_analyses):
        ic("Starting analysis for experiment", i)
        ic(analysis.topk_inputs_decoded(experiment_data.tokenizer, k=10))
        ic(analysis.outputs_for_topk_inputs_decoded(experiment_data, k_inputs=10, k_outputs=3))
        ic(analysis.outputs_for_topk_inputs_decoded(experiment_data, k_inputs=10, k_outputs=3, circuit=False))
        ic(analysis.loss_for_topk_inputs(experiment_data, k=10))
        # TODO: loss for convex combination

        ic("Now for comparison, using the initial coefficients:")
        # What are the results for the initial coefficients?
        ic(analysis.topk_inputs_decoded(experiment_data.tokenizer, k=10, final_coefficients=False))
        ic(
            analysis.outputs_for_topk_inputs_decoded(
                experiment_data, k_inputs=10, k_outputs=3, circuit=True, final_coefficients=False
            )
        )
        ic(
            analysis.outputs_for_topk_inputs_decoded(
                experiment_data, k_inputs=10, k_outputs=3, circuit=False, final_coefficients=False
            )
        )


if __name__ == "__main__":
    main()
