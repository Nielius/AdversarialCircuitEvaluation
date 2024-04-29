import shlex

from acdc.nudb.adv_opt.data_fetchers import AdvOptTaskName


def command(random_seed: int, task: AdvOptTaskName) -> str:
    return shlex.join(
        [
            "python",
            "-m",
            "acdc.nudb.adv_opt.main",
            f"task={task.lower()}",
            "num_epochs=500",
            "use_wandb=true",
            "coefficient_renormalization=halving",
            "wandb_project_name=test-halving",
            f"wandb_run_name={task.lower()}-halving-{random_seed}",
            f"wandb_group_name={task.lower()}-halving",
            f"random_seed={random_seed}",
        ]
    )
