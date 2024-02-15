# from experiments.launcher import KubernetesJob, WandbIdentifier, launch
import random
import subprocess
import sys
from typing import List

import numpy as np

METRICS_FOR_TASK = {
    "ioi": ["kl_div", "logit_diff"],
    "tracr-reverse": ["l2"],
    "tracr-proportion": ["kl_div", "l2"],
    "induction": ["kl_div", "nll"],
    "docstring": ["kl_div", "docstring_metric"],
    "greaterthan": ["kl_div", "greaterthan"],
}


def make_yamls(
    TASKS: list[str],
    testing: bool,
    reset_networks: bool,
    template_filename: str,
    container: str,
) -> list[str]:
    """
    Takes a yaml file template and creates many yaml files to pass to kubectl.
    """
    NUM_SPACINGS = 5 if reset_networks else 21
    expensive_base_regularization_params = np.concatenate(
        [
            10 ** np.linspace(-2, 0, 11),
            np.linspace(1, 10, 10)[1:],
            np.linspace(10, 250, 13)[1:],
        ]
    )

    if reset_networks:
        base_regularization_params = 10 ** np.linspace(-2, 1.5, NUM_SPACINGS)
    else:
        base_regularization_params = expensive_base_regularization_params

    seed = 1507014021
    random.seed(seed)

    yamls: List[List[str]] = []
    i = 0
    for reset_network in [int(reset_networks)]:
        for zero_ablation in [0, 1]:
            for task in TASKS:
                for metric in METRICS_FOR_TASK[task]:
                    if task.startswith("tracr"):
                        # Typical metric value range: 0.0-0.1
                        regularization_params = 10 ** np.linspace(-3, 0, 11)

                        if task == "tracr-reverse":
                            num_examples = 6
                            seq_len = 5
                        elif task == "tracr-proportion":
                            num_examples = 50
                            seq_len = 5
                        else:
                            raise ValueError("Unknown task")

                    elif task == "greaterthan":
                        if metric == "kl_div":
                            # Typical metric value range: 0.0-20
                            regularization_params = base_regularization_params
                        elif metric == "greaterthan":
                            # Typical metric value range: -1.0 - 0.0
                            regularization_params = 10 ** np.linspace(-4, 2, NUM_SPACINGS)
                        else:
                            raise ValueError("Unknown metric")
                        num_examples = 100
                        seq_len = -1
                    elif task == "docstring":
                        seq_len = 41
                        if metric == "kl_div":
                            # Typical metric value range: 0.0-10.0
                            regularization_params = expensive_base_regularization_params
                        elif metric == "docstring_metric":
                            # Typical metric value range: -1.0 - 0.0
                            regularization_params = 10 ** np.linspace(-4, 2, 21)
                        else:
                            raise ValueError("Unknown metric")
                        num_examples = 50
                    elif task == "ioi":
                        num_examples = 100
                        seq_len = -1
                        if metric == "kl_div":
                            # Typical metric value range: 0.0-12.0
                            regularization_params = base_regularization_params
                        elif metric == "logit_diff":
                            # Typical metric value range: -0.31 -- -0.01
                            regularization_params = 10 ** np.linspace(-4, 2, NUM_SPACINGS)
                        else:
                            raise ValueError("Unknown metric")
                    elif task == "induction":
                        seq_len = 300
                        num_examples = 50
                        if metric == "kl_div":
                            # Typical metric value range: 0.0-16.0
                            regularization_params = expensive_base_regularization_params
                        elif metric == "nll":
                            # Typical metric value range: 0.0-16.0
                            regularization_params = expensive_base_regularization_params
                        else:
                            raise ValueError("Unknown metric")
                    else:
                        raise ValueError("Unknown task")

                    for lambda_reg in [0.01] if testing else regularization_params:
                        # should do this simpler way
                        wandb_project = "subnetwork-probing"
                        wandb_entity = "tkwa-team"
                        wandb_group = "edge_sp_group_2"
                        wandb_name = (
                            f"tkwa-sp-{task}-{i:05d}{'-optional' if task in ['induction', 'docstring'] else ''}"
                        )

                        command = [
                            "python",
                            "subnetwork_probing/train_edge_sp.py",
                            f"--task={task}",
                            f"--lambda-reg={lambda_reg:.3f}",
                            f"--wandb-name={wandb_name}",
                            f"--wandb-project={wandb_project}",
                            f"--wandb-entity={wandb_entity}",
                            f"--wandb-group={wandb_group}",
                            f"--device={'cpu' if task.startswith('tracr') else 'cuda'}",
                            f"--epochs={1 if testing else 10000}",
                            f"--zero-ablation={zero_ablation}",
                            f"--reset-subject={reset_network}",
                            f"--seed={random.randint(0, 2**32 - 1)}",
                            f"--loss-type={metric}",
                            f"--num-examples={6 if testing else num_examples}",
                            f"--seq-len={seq_len}",
                            f"--n-loss-average-runs={1 if testing else 20}",
                            "--wandb-dir=/tmp/training",  # If it doesn't exist wandb will use /tmp
                            f"--wandb-mode={'offline' if testing else 'online'}",
                            f"--torch-num-threads={4}",
                        ]
                        if i == 0:
                            print(" ".join(command))

                        template = open(template_filename).read()
                        yaml = template.format(
                            COMMAND=" ".join(command),
                            CONTAINER=container,
                            NAME=f"sp-{task}-{i:05d}",
                            WANDB_GROUP=wandb_group,
                            WANDB_PROJECT=wandb_project,
                            WANDB_JOB_NAME=wandb_name,
                            WANDB_ENTITY=wandb_entity,
                            LAUNCH_ID=f"sp-{task}-{i:05d}",
                            COMMIT_HASH=subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True)
                            .stdout.decode()
                            .strip(),
                            CPU=4,
                            MEMORY="16Gi",
                            GPU=0 if task.startswith("tracr") else 1,
                            OMP_NUM_THREADS="'4'",  # is this right?
                        )

                        yamls.append(yaml)
                        i += 1
    return yamls


if __name__ == "__main__":
    for reset_networks in [False]:
        tasks = ["tracr-proportion"]
        yamls_list = make_yamls(
            tasks,
            testing=False,
            reset_networks=reset_networks,
            template_filename="subnetwork_probing/runner_template.yaml",
            container="ghcr.io/tkwa/subnetwork_probing:v1",
        )
        if "--launch-one" in sys.argv:
            yamls_list = yamls_list[:1]
            print(yamls_list[0])
        yamls_for_all_jobs = "\n\n---\n\n".join(yamls_list)
        if not any(s in sys.argv for s in ["--dryrun", "--dry-run", "-d"]):
            print(f"Launching {len(yamls_list)} jobs")
            subprocess.run(
                ["kubectl", "create", "-f", "-"],
                check=True,
                input=yamls_for_all_jobs.encode(),
            )
