import logging
import random
from pathlib import Path

import hydra
import hydra.core.hydra_config as hydra_config
import omegaconf
import torch
import torch.nn.functional as F
import wandb
from hydra.core.config_store import ConfigStore
from jaxtyping import Integer
from torch.optim.lr_scheduler import StepLR

from acdc.nudb.adv_opt.data_fetchers import (
    AdvOptExperimentData,
    AdvOptTaskName,
    get_standard_experiment_data,
)
from acdc.nudb.adv_opt.noise_generators import (
    NoNoiseGenerator,
    SPNoiseGenerator,
    ClampedSPNoiseGenerator,
    IntermittentNoiseGenerator,
    ScheduledNoiseGenerator,
)
from acdc.nudb.adv_opt.settings import (
    ExperimentArtifacts,
    ExperimentSettings,
    TaskSpecificSettings,
    TracrReverseTaskSpecificSettings,
)
from acdc.nudb.adv_opt.utils import device, joblib_memory

logger = logging.getLogger(__name__)

cs = ConfigStore.instance()
cs.store(name="config_schema", node=ExperimentSettings)
# Not sure if/why you need to add the two lines below. Maybe just for checking that config schema?
# (I'm now 80% sure that these things are only for schema matching.)
cs.store(group="task", name="greaterthan_schema", node=TaskSpecificSettings)
cs.store(group="task", name="tracr_reverse_schema", node=TracrReverseTaskSpecificSettings)


@joblib_memory.cache
def get_experiment_data_cached(task_name: AdvOptTaskName) -> AdvOptExperimentData:
    return get_standard_experiment_data(task_name)


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(settings: ExperimentSettings) -> None:
    if settings.random_seed is not None:
        torch.manual_seed(settings.random_seed)
        random.seed(settings.random_seed)

    output_base_dir = Path(hydra_config.HydraConfig.get().runtime.output_dir)
    artifacts = ExperimentArtifacts()

    # Log in to your W&B account
    wandb.login()
    wandb.init(
        project=settings.wandb_project_name or "advopt-input-and-patch",
        config=omegaconf.OmegaConf.to_container(settings, resolve=True),
        mode="online" if settings.use_wandb else "disabled",
        name=settings.wandb_run_name,
        group=settings.wandb_group_name,
        tags=settings.wandb_tags,
    )

    experiment_data = (
        get_experiment_data_cached(task_name=settings.task.task_name)
        if settings.use_experiment_cache
        else get_standard_experiment_data(task_name=settings.task.task_name)
    )
    experiment_data.masked_runner.masked_transformer.freeze_weights()

    # stack test_data and validation_data
    base_input: Integer[torch.Tensor, "batch pos"] = torch.cat(
        [experiment_data.task_data.test_data, experiment_data.task_data.validation_data]
    )
    base_input_embedded = experiment_data.masked_runner.masked_transformer.model.embed(base_input)
    artifacts.base_input = base_input

    base_patch_input: Integer[torch.Tensor, "batch pos"] = torch.cat(
        [experiment_data.task_data.test_patch_data, experiment_data.task_data.validation_patch_data]
    )
    base_patch_input_embedded = experiment_data.masked_runner.masked_transformer.model.embed(base_patch_input)
    artifacts.base_patch_input = base_patch_input

    dummy_input = experiment_data.task_data.test_data[0, ...]

    convex_coefficients = torch.rand(base_input.shape[0], requires_grad=True, device=device)
    convex_coefficients_patch = torch.rand(base_input.shape[0], requires_grad=True, device=device)
    if settings.optimization_method == "adam":
        optimizer = torch.optim.Adam([convex_coefficients, convex_coefficients_patch], lr=settings.adam_lr)
    elif settings.optimization_method == "adamw":
        optimizer = torch.optim.AdamW([convex_coefficients, convex_coefficients_patch], lr=settings.adam_lr)
    else:
        raise ValueError(f"Unknown optimization method: {settings.optimization_method}")

    match settings.adam_lr_schedule:
        case "constant":
            scheduler = None
        case "step_increase":
            scheduler = StepLR(optimizer, step_size=100, gamma=2)
        case "step_decrease":
            scheduler = StepLR(optimizer, step_size=int(0.1 * settings.num_epochs), gamma=0.5)
        case _:
            raise NotImplementedError(f"Unknown learning rate schedule: {settings.adam_lr_schedule}")

    artifacts.coefficients_init = convex_coefficients.detach().clone()
    artifacts.coefficients_init_patch = convex_coefficients_patch.detach().clone()

    temperature = 1.0

    if settings.task.task_name == AdvOptTaskName.TRACR_REVERSE:
        settings_ = omegaconf.OmegaConf.to_object(settings)
        assert isinstance(settings_.task, TracrReverseTaskSpecificSettings)
        if settings_.task.artificially_corrupt_model:
            # in this case, the model is perfect, so we can't do any optimization
            # for testing purposes, let's make it less perfect by changing the weights of the last MLP
            torch.nn.init.normal_(experiment_data.masked_runner.masked_transformer.model.blocks[3].mlp.W_in)
            torch.nn.init.normal_(experiment_data.masked_runner.masked_transformer.model.blocks[3].mlp.W_out)

    noise_generator = _get_noise_generator(settings.noise_schedule, convex_coefficients.shape)

    for i in range(settings.num_epochs):
        optimizer.zero_grad()
        noise, noise_patch = noise_generator.generate_noise(i, settings.num_epochs)
        convex_coefficients_with_noise_and_temp = (convex_coefficients + noise) / temperature
        convex_coefficients_patch_with_noise_and_temp = (convex_coefficients_patch + noise_patch) / temperature

        circuit_output = experiment_data.masked_runner.run_with_linear_combination_of_input_and_patch(
            input_embedded=base_input_embedded,
            patch_input_embedded=base_patch_input_embedded,
            dummy_input=dummy_input,
            coefficients=convex_coefficients_with_noise_and_temp,
            coefficients_patch=convex_coefficients_patch_with_noise_and_temp,
            edges_to_ablate=list(experiment_data.ablated_edges),
        )
        full_output = experiment_data.masked_runner.run_with_linear_combination_of_input_and_patch(
            input_embedded=base_input_embedded,
            patch_input_embedded=base_patch_input_embedded,
            dummy_input=dummy_input,
            coefficients=convex_coefficients_with_noise_and_temp,
            coefficients_patch=convex_coefficients_patch_with_noise_and_temp,
            edges_to_ablate=[],
        )
        negative_loss = -1 * experiment_data.loss_fn(
            circuit_output,
            full_output,
        )
        negative_loss.backward()
        with torch.no_grad():
            wandb.log(
                {
                    "loss": -1 * negative_loss,
                    "temperature": temperature,
                    "epoch": i,
                    "lr": optimizer.param_groups[0]["lr"],
                    "coefficients_entropy": torch.distributions.Categorical(logits=convex_coefficients)
                    .entropy()
                    .item(),
                    "coefficients_patch_entropy": torch.distributions.Categorical(logits=convex_coefficients_patch)
                    .entropy()
                    .item(),
                    "gradient_norm": torch.norm(convex_coefficients.grad).item(),
                    "coefficients_norm": torch.norm(convex_coefficients).item(),
                    "noise_kl_div": F.kl_div(
                        convex_coefficients,
                        convex_coefficients_with_noise_and_temp * temperature,
                        reduction="none",
                        log_target=True,
                    ).sum(),
                }
            )

        optimizer.step()

        match settings.temperature_schedule:
            case "stable":  # this is stable, then drops linearly in last 80%
                temperature = min(1.0, 1e-4 + 5 * (1 - (i / settings.num_epochs)))
            case "stable_low":  # this is stable (but low), then drops linearly in last 80%
                base_temperature = 0.8
                temperature = min(base_temperature, 5 * base_temperature * (1 - (i / settings.num_epochs)) + 1e-4)
            case "constant_low":
                temperature = 0.8
            case "linear":
                temperature = 0.0001 + 1 - i / settings.num_epochs

        if scheduler is not None:
            scheduler.step()
            # scheduler.step(-1 * negative_loss) # if we're using plateau
        logger.info(f"Epoch {i}, loss: {-1 * negative_loss.item()}")
        logger.debug("Convex coefficients:", convex_coefficients)
        # wandb.log({"loss": loss, "convex_combination": convex_coefficients})

    artifacts.coefficients_final = convex_coefficients.detach().clone()
    artifacts.coefficients_final_patch = convex_coefficients_patch.detach().clone()
    artifacts.save(output_base_dir / "artifacts")
    logger.info("Finished training. Output stored in %s", output_base_dir)

    wandb.finish()


def _get_noise_generator(noise_schedule: str, shape: tuple[int]) -> ScheduledNoiseGenerator:
    match noise_schedule:
        case "absent":
            return NoNoiseGenerator(shape=shape, device=device)
        case "constant":
            return SPNoiseGenerator(shape=shape, device=device)
        case "clamped":
            return ClampedSPNoiseGenerator(shape=shape, device=device, max=10, scaling=0.1)
        case "intermittent_constant":
            return IntermittentNoiseGenerator(
                SPNoiseGenerator(shape=shape, device=device),
                shape=shape,
                noise_epoch_length=20,
                no_noise_epoch_length=20,
                device=device,
            )
        case "intermittent_clamped":
            return IntermittentNoiseGenerator(
                ClampedSPNoiseGenerator(shape=shape, device=device, max=10, scaling=0.1),
                shape=shape,
                noise_epoch_length=20,
                no_noise_epoch_length=20,
                device=device,
            )
        case _:
            raise NotImplementedError(f"Unknown noise schedule: {noise_schedule}")


if __name__ == "__main__":
    main()
