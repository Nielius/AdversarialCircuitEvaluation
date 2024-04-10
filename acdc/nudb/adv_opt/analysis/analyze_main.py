from pathlib import Path

import torch
import yaml

from acdc.nudb.adv_opt.data_fetchers import AdvOptTaskName, get_standard_experiment_data
from acdc.nudb.adv_opt.settings import ExperimentArtifacts
from acdc.nudb.adv_opt.utils import deep_map

base_dir = Path("/home/niels/proj/mats/data/sync-2024-02-27T09_50_25/2024-02-26/18-38-25")
artifact_dir = base_dir / "artifacts"


# Load hydra config
hydra_config_file = base_dir / ".hydra" / "config.yaml"
hydra_config = yaml.safe_load((hydra_config_file).read_text())

task_name = AdvOptTaskName[hydra_config["task"]["task_name"]]

experiment_data = get_standard_experiment_data(task_name)

artifacts = ExperimentArtifacts.load(artifact_dir)

# Print the topk inputs that break the circuit the most
coeffs = artifacts.coefficients_final
assert coeffs is not None
base_input = artifacts.base_input
assert base_input is not None
topk = torch.topk(coeffs, 10)
tokenizer = experiment_data.masked_runner.masked_transformer.model.tokenizer
assert tokenizer is not None
for token in base_input[topk.indices, :]:
    print(tokenizer.decode(token))

# What are the predicted outputs for the circuit and for the model for tis

tokenizer.decode(base_input[topk.indices, :])

# Note: this patch_input is likely to change once we start optimizing over corrupted data.
# But the current version of main.py just uses the first patch input from the test data.
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

# Refactor?
#
# - utility function to map from pytorch to list of lists
# - very similar output and decoding happening here

# For each of the top 10 worst input points,
# get the topk most likely outputs from the circuit and the full model
# and tokenize-decode them.
topk_outputs_circuit = torch.topk(outputs_from_circuit[:, -1, :], k=10)
topk_output_circuit_decoded = [
    [tokenizer.decode(output) for output in outputs] for outputs in topk_outputs_circuit.indices
]
topk_outputs_full_model = torch.topk(outputs_from_full_model[:, -1, :], k=10)
topk_output_full_model_decoded = [
    [tokenizer.decode(output) for output in outputs] for outputs in topk_outputs_full_model.indices
]

output_circuit_for_convex_combination = experiment_data.masked_runner.run_with_linear_combination(
    input_embedded=experiment_data.masked_runner.masked_transformer.model.embed(base_input),
    dummy_input=experiment_data.task_data.test_data[0, ...],
    coefficients=coeffs,
    patch_input=patch_input,
    edges_to_ablate=list(experiment_data.ablated_edges),
)
output_full_model_for_convex_combination = experiment_data.masked_runner.run_with_linear_combination(
    input_embedded=experiment_data.masked_runner.masked_transformer.model.embed(base_input),
    dummy_input=experiment_data.task_data.test_data[0, ...],
    coefficients=coeffs,
    patch_input=patch_input,
    edges_to_ablate=[],
)

loss_convex_combination = experiment_data.loss_fn(
    output_circuit_for_convex_combination,
    output_full_model_for_convex_combination,
)
topk_output_circuit_convex_combination = torch.topk(output_circuit_for_convex_combination[:, -1, :], k=10)
topk_output_full_model_convex_combination = torch.topk(output_full_model_for_convex_combination[:, -1, :], k=10)
topk_output_circuit_convex_combination_decoded = [
    [tokenizer.decode(output) for output in outputs] for outputs in topk_output_circuit_convex_combination.indices
]
topk_output_full_model_convex_combination_decoded = [
    [tokenizer.decode(output) for output in outputs] for outputs in topk_output_full_model_convex_combination.indices
]


# What do we want now?
# - run circuit on inputs
# - run full model on inputs
# - compare outputs
# - interpret
# - calculate loss


t = topk_outputs_full_model.indices

type(t.tolist())
type(t.tolist()[0])

list(map(lambda x: tokenizer.decode(x), t.tolist()))

deep_map(lambda x: tokenizer.decode(x), t.tolist())
