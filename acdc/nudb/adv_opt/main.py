import logging
import random
from pathlib import Path

import hydra
import hydra.core.hydra_config as hydra_config
import omegaconf
import torch
from hydra.core.config_store import ConfigStore
from jaxtyping import Integer

import wandb
from acdc.nudb.adv_opt.data_fetchers import (
    AdvOptExperimentData,
    AdvOptTaskName,
    get_standard_experiment_data,
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
        project="advopt-input-and-patch",
        config=omegaconf.OmegaConf.to_container(settings, resolve=True),
        mode="online" if settings.use_wandb else "disabled",
        name=settings.wandb_run_name,
        group=settings.wandb_group_name,
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
    artifacts.coefficients_init = convex_coefficients.detach().clone()
    artifacts.coefficients_init_patch = convex_coefficients_patch.detach().clone()

    if settings.task.task_name == AdvOptTaskName.TRACR_REVERSE:
        settings_ = omegaconf.OmegaConf.to_object(settings)
        assert isinstance(settings_.task, TracrReverseTaskSpecificSettings)
        if settings_.task.artificially_corrupt_model:
            # in this case, the model is perfect, so we can't do any optimization
            # for testing purposes, let's make it less perfect by changing the weights of the last MLP
            torch.nn.init.normal_(experiment_data.masked_runner.masked_transformer.model.blocks[3].mlp.W_in)
            torch.nn.init.normal_(experiment_data.masked_runner.masked_transformer.model.blocks[3].mlp.W_out)

    for i in range(settings.num_epochs):
        # normalize linear coefficients
        # with torch.no_grad():
        #     # I suppose the no_grad is necessary here?
        #     # layer norm is usually involved in the gradient, isn't it? so maybe we should just keep it
        #     # except of course for the first iteration?
        #     # normalize so that it sums to 1
        #     convex_coefficients.divide /= convex_coefficients / convex_coefficients.sum()
        # convex_coefficients.requires_grad = True
        #
        # Normalizing has no effect, because the softmax implementation subtracts the maximum anyway.

        # Which optimizer would be good for this? Try Adam or AdamW for now
        # Normalization of the coefficients?
        # Try different initizalizations of the coefficients, see if it's sensitive to that
        # maybe just set all to zero in initialization?
        optimizer.zero_grad()
        circuit_output = experiment_data.masked_runner.run_with_linear_combination_of_input_and_patch(
            input_embedded=base_input_embedded,
            patch_input_embedded=base_patch_input_embedded,
            dummy_input=dummy_input,
            coefficients=convex_coefficients,
            coefficients_patch=convex_coefficients_patch,
            edges_to_ablate=list(experiment_data.ablated_edges),
        )
        full_output = experiment_data.masked_runner.run_with_linear_combination_of_input_and_patch(
            input_embedded=base_input_embedded,
            patch_input_embedded=base_patch_input_embedded,
            dummy_input=dummy_input,
            coefficients=convex_coefficients,
            coefficients_patch=convex_coefficients_patch,
            edges_to_ablate=[],
        )
        negative_loss = -1 * experiment_data.loss_fn(
            circuit_output,
            full_output,
        )
        negative_loss.backward()
        optimizer.step()
        wandb.log({"loss": -1 * negative_loss})
        logger.info(f"Epoch {i}, loss: {-1 * negative_loss.item()}")
        logger.debug("Convex coefficients:", convex_coefficients)
        # wandb.log({"loss": loss, "convex_combination": convex_coefficients})

    artifacts.coefficients_final = convex_coefficients.detach().clone()
    artifacts.coefficients_final_patch = convex_coefficients_patch.detach().clone()
    artifacts.save(output_base_dir / "artifacts")
    logger.info("Finished training. Output stored in %s", output_base_dir)

    wandb.finish()


if __name__ == "__main__":
    main()
