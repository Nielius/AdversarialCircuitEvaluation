import random
from dataclasses import asdict, dataclass

import torch
from jaxtyping import Integer

import wandb
from acdc.nudb.adv_opt.data_fetchers import EXPERIMENT_DATA_PROVIDERS, AdvOptTaskName
from acdc.nudb.adv_opt.main_circuit_performance_distribution import kl_div_on_output_logits

# Log in to your W&B account
wandb.login()

device = "cuda:0" if torch.cuda.is_available() else "cpu"


@dataclass
class ExperimentSettings:
    task_name: AdvOptTaskName
    num_epochs: int
    metric_name: str = "kl_div"
    random_seed: int | None = None


settings = ExperimentSettings(
    task_name=AdvOptTaskName.TRACR_REVERSE,
    metric_name="l2",
    num_epochs=5,
)


if settings.random_seed is not None:
    torch.manual_seed(settings.random_seed)
    random.seed(settings.random_seed)


wandb.init(
    project="pytorch-intro",
    config=asdict(settings),
    mode="disabled",
)

experiment_data = EXPERIMENT_DATA_PROVIDERS[AdvOptTaskName.TRACR_REVERSE].get_experiment_data(
    num_examples=30,
    metric_name="l2",
    device=device,
)

# stack test_data and validation_data
base_input: Integer[torch.Tensor, "batch pos"] = torch.cat(
    [experiment_data.task_data.test_data, experiment_data.task_data.validation_data]
)

experiment_data.masked_runner.masked_transformer.freeze_weights()


convex_coefficients = torch.rand(base_input.shape[0], requires_grad=True)


# maybe should do some kind of normalization at the end of

# TODO: can probably choose better optimizers?
optimizer = torch.optim.Adam([convex_coefficients], lr=1e-3)

base_input_embedded = experiment_data.masked_runner.masked_transformer.model.embed(base_input)


dummy_input = experiment_data.task_data.test_data[
    0, ...
]  # this is dummy input, because the embed hook replaces it anyway. But its shape is used
patch_input = experiment_data.task_data.test_patch_data[0, ...]


for i in range(settings.num_epochs):
    # normalize linear coefficients
    # with torch.no_grad():
    #     # I suppose the no_grad is necessary here?
    #     # layer norm is usually involved in the gradient, isn't it? so maybe we should just keep it
    #     # except of course for the first iteration?
    #     # normalize so that it sums to 1
    #     convex_coefficients.divide /= convex_coefficients / convex_coefficients.sum()
    # convex_coefficients.requires_grad = True

    # Which optimizer would be good for this? Try Adam or AdamW for now
    # Normalization of the coefficients?
    # Try different initizalizations of the coefficients, see if it's sensitive to that
    # maybe just set all to zero in initialization?
    optimizer.zero_grad()
    circuit_output = experiment_data.masked_runner.run_with_linear_combination(
        input_embedded=base_input_embedded,
        dummy_input=dummy_input,
        coefficients=convex_coefficients,
        patch_input=patch_input,
        edges_to_ablate=list(experiment_data.ablated_edges),
    )
    full_output = experiment_data.masked_runner.run_with_linear_combination(
        input_embedded=base_input_embedded,
        dummy_input=dummy_input,
        coefficients=convex_coefficients,
        patch_input=patch_input,
        edges_to_ablate=[],
    )
    negative_loss = -1 * kl_div_on_output_logits(
        circuit_output,
        full_output,
        last_sequence_position_only=experiment_data.metric_last_sequence_position_only,
    )
    negative_loss.backward()
    optimizer.step()
    wandb.log({"loss": -1 * negative_loss})
    print(f"Epoch {i}, loss: {-1 * negative_loss.item()}")
    print("Convex coefficients:", convex_coefficients)
    # TODO: should normalize the coefficients somewhere around here
    # wandb.log({"loss": loss, "convex_combination": convex_coefficients})
