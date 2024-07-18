"""
NOTE: I think this was a draft file, and a2024_02_28_input_only.py is the final version.
"""


from dataclasses import dataclass
from pathlib import Path

import torch
import yaml
from jaxtyping import Float

from acdc.nudb.adv_opt.data_fetchers import AdvOptExperimentData, AdvOptTaskName, get_standard_experiment_data
from acdc.nudb.adv_opt.settings import ExperimentArtifacts
from acdc.nudb.adv_opt.utils import deep_map

base_dir = Path("/home/niels/proj/mats/data/outputs/input_only")

# useful experiments to look at
# All have the same Adam LR of 0.1, and 400 epochs, but different random seeds
greaterthan_experiment_paths = [
    "2024-02-28-051332-greaterthan_seed_4321/4321",
    "2024-02-28-051332-greaterthan_seed_4322/4322",
    "2024-02-28-051332-greaterthan_seed_4323/4323",
    "2024-02-28-051332-greaterthan_seed_4324/4324",
    "2024-02-28-051332-greaterthan_seed_4325/4325",
    "2024-02-28-051332-greaterthan_seed_4326/4326",
    "2024-02-28-051332-greaterthan_seed_4327/4327",
    "2024-02-28-051332-greaterthan_seed_4328/4328",
    "2024-02-28-051332-greaterthan_seed_4329/4329",
    "2024-02-28-051332-greaterthan_seed_4330/4330",
]
ioi_experiment_paths = [
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
]


@dataclass
class ExperimentAnalysis:
    """WIP, might be nicer"""

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
        return torch.topk(coeffs, k)

    def topk_inputs_decoded(self, tokenizer, k: int = 10, final_coefficients: bool = True) -> list[str]:
        topk = self.topk_inputs(k, final_coefficients)
        return [tokenizer.decode(token) for token in self.artifacts.base_input[topk.indices, :]]

    def topk_input_score(
        self, experiment_data: AdvOptExperimentData, k: int = 10, final_coefficients: bool = True
    ) -> Float[torch.Tensor, " batch"]:
        topk = self.topk_inputs(k, final_coefficients)
        return self.artifacts.coefficients_final[topk.indices]

    def outputs_for_topk_inputs(
        self, experiment_data: AdvOptExperimentData, k: int = 10, circuit: bool = True, final_coefficients: bool = True
    ) -> Float[torch.Tensor, "batch pos vocab"]:
        """Get output for full model if `circuit` is set to False."""
        topk = self.topk_inputs(k, final_coefficients)
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
    ) -> Float[torch.Tensor, "batch pos vocab"]:
        """For each of the k_inputs worst inputs, get the k_outputs most likely output values, decoded."""
        outputs = self.outputs_for_topk_inputs(experiment_data, k_inputs, circuit)
        return deep_map(experiment_data.tokenizer.decode, torch.topk(outputs[:, -1, :], k=k_outputs).indices.tolist())

    def loss_for_topk_inputs(
        self,
        experiment_data: AdvOptExperimentData,
        k: int = 10,
    ) -> Float[torch.Tensor, " batch"]:
        outputs_circuit = self.outputs_for_topk_inputs(experiment_data, k, circuit=True)
        outputs_full_model = self.outputs_for_topk_inputs(experiment_data, k, circuit=False)
        return experiment_data.loss_fn(outputs_circuit, outputs_full_model)


ioi_experiment_analyses = [
    ExperimentAnalysis.from_dir(base_dir / experiment_dir) for experiment_dir in ioi_experiment_paths
]

experiment_data = ioi_experiment_analyses[0].get_experiment_data()

convex_combinations_tensor = torch.stack(
    [analysis.artifacts.convex_combination(experiment_data.embedder) for analysis in ioi_experiment_analyses]
)

analysis = ioi_experiment_analyses[0]
analysis.topk_inputs_decoded(experiment_data.tokenizer, k=10)
analysis.outputs_for_topk_inputs_decoded(experiment_data, k_inputs=10, k_outputs=2)

outputs = analysis.outputs_for_topk_inputs(experiment_data, k=10)

outputs_flattened = outputs.flatten(start_dim=1, end_dim=2)
torch.cdist(outputs_flattened, outputs_flattened)


[analysis.topk_inputs_decoded(experiment_data.tokenizer, k=10) for analysis in ioi_experiment_analyses]

for analysis in ioi_experiment_analyses:
    topk = torch.topk(analysis.artifacts.coefficients_init, 10)
    tokenizer = experiment_data.tokenizer
    assert tokenizer is not None
    assert analysis.artifacts.base_input is not None
    for token in analysis.artifacts.base_input[topk.indices, :]:
        print(tokenizer.decode(token))


# Test if they are different or not
flattened = torch.flatten(convex_combinations_tensor, start_dim=1, end_dim=2)
dist = torch.cdist(flattened, flattened)
print(dist)

num_of_top = 10


# Decode topk
# Calculate loss of topk
# Calculate loss of convex combination


# ------ old code -------


def get_embedder() -> torch.nn.Module:
    experiment_data = get_standard_experiment_data(task_name=AdvOptTaskName.GREATERTHAN)
    return experiment_data.masked_runner.masked_transformer.model.embed


def load_hydra_config(experiment_dir: Path) -> dict:
    hydra_config_file = experiment_dir / ".hydra" / "config.yaml"
    return yaml.safe_load((hydra_config_file).read_text())


def load_convex_combination(experiment_dir: str, embedder: torch.nn.Module) -> Float[torch.Tensor, "batch vocab"]:
    artifacts = ExperimentArtifacts.load(base_dir / experiment_dir / "artifacts")
    return artifacts.convex_combination(embedder)


def load_final_coefficients(experiment_dir: str) -> Float[torch.Tensor, "batch vocab"]:
    final_coefficients = ExperimentArtifacts.load(base_dir / experiment_dir / "artifacts").coefficients_final
    assert final_coefficients is not None

    return final_coefficients


def load_initial_coefficients(experiment_dir: str) -> Float[torch.Tensor, "batch vocab"]:
    initial_coefficients = ExperimentArtifacts.load(base_dir / experiment_dir / "artifacts").coefficients_init
    assert initial_coefficients is not None

    return initial_coefficients


embedder = get_embedder()
convex_combinations_greaterthan = [
    load_convex_combination(experiment_dir, embedder) for experiment_dir in greaterthan_experiment_paths
]
convex_combinations_ioi = [load_convex_combination(experiment_dir, embedder) for experiment_dir in ioi_experiment_paths]

# --------------

# Turn convex_combinations_greaterthan into a tensor
convex_combinations_tensor = torch.stack(convex_combinations_greaterthan).flatten(start_dim=1, end_dim=2)
dist = torch.cdist(convex_combinations_tensor, convex_combinations_tensor)


convex_combinations_tensor = torch.stack(convex_combinations_ioi).flatten(start_dim=1, end_dim=2)
dist = torch.cdist(convex_combinations_tensor, convex_combinations_tensor)

# --------------


torch.flatten(convex_combinations_tensor, start_dim=1, end_dim=2).shape
convex_combinations_greaterthan[0].shape
convex_combinations_t = torch.tensor(convex_combinations_greaterthan)

print(123)

coefficients_greaterthan = torch.stack(
    [load_final_coefficients(experiment_dir) for experiment_dir in greaterthan_experiment_paths]
)
dist = torch.cdist(coefficients_greaterthan, coefficients_greaterthan)


coefficients_ioi = torch.stack([load_final_coefficients(experiment_dir) for experiment_dir in ioi_experiment_paths])
dist = torch.cdist(coefficients_ioi, coefficients_ioi)


coefficients_greaterthan = torch.stack(
    [load_initial_coefficients(experiment_dir) for experiment_dir in greaterthan_experiment_paths]
)
dist = torch.cdist(coefficients_greaterthan, coefficients_greaterthan)


coefficients_ioi = torch.stack([load_initial_coefficients(experiment_dir) for experiment_dir in ioi_experiment_paths])
dist = torch.cdist(coefficients_ioi, coefficients_ioi)

coefficients_greaterthan.shape
