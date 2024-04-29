import subprocess
import time
from pathlib import Path
from typing import Annotated

import typer

from acdc.nudb.adv_opt.data_fetchers import AdvOptTaskName

MEMORY_USAGE = {
    AdvOptTaskName.DOCSTRING: "2Gi",
    AdvOptTaskName.IOI: "3Gi",
    AdvOptTaskName.GREATERTHAN: "3Gi",
}


def construct_yaml(
    job_name: str, command: str, commit_hash: str = "main", task: AdvOptTaskName = AdvOptTaskName.DOCSTRING
) -> str:
    runner_template = (Path(__file__).parent / "runner.yaml").read_text()

    return runner_template.format(
        NAME=job_name,
        PRIORITY="normal-batch",
        COMMAND=command,
        CONTAINER_TAG="latest",
        COMMIT_HASH=commit_hash,
        CPU="1",
        MEMORY=MEMORY_USAGE.get(task, "2Gi"),
        GPU="1",
    )


# create a Typer app
app = typer.Typer()


@app.command()
def print(
    job_name: str,
    command: str,
):
    typer.echo(construct_yaml(job_name, command))


@app.command()
def launch(
    job_name: str,
    command: str,
    commit_hash: Annotated[str, typer.Option("--commit", "-c")] = "main",
    task: AdvOptTaskName = AdvOptTaskName.DOCSTRING,
    should_follow_logs: bool = True,
):
    subprocess.run(
        ["kubectl", "create", "-f", "-"],
        check=True,
        input=construct_yaml(job_name, command, commit_hash=commit_hash, task=task).encode(),
    )

    typer.echo("Job launched.")
    if should_follow_logs:
        follow_logs(job_name)


def get_pod_name_from_job_name(job_name: str) -> str:
    return subprocess.run(
        f"kubectl get pod --selector=job-name={job_name} -o name",
        shell=True,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def follow_logs(job_name: str):
    typer.echo("Starting to follow logs...")
    while True:
        try:
            pod_name = get_pod_name_from_job_name(job_name)
            break
        except subprocess.CalledProcessError:
            typer.echo("No pod found yet. Retrying in 5 seconds...")
            time.sleep(5)
    typer.echo(f"Found pod {pod_name}.")
    while True:
        pod_phase = subprocess.run(
            f"kubectl get {pod_name} --template '{{{{.status.phase}}}}'",
            shell=True,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

        if pod_phase == "Completed" or pod_phase == "Running":
            break
        else:
            typer.echo(f"Pod phase is {pod_phase}. Retrying in 5 seconds...")
        time.sleep(5)
    typer.echo("Logs:")
    subprocess.run(
        f"kubectl logs -f {pod_name}",
        shell=True,
        check=True,
    )


# run the app
if __name__ == "__main__":
    app()
