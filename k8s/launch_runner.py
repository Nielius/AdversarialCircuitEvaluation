import subprocess
from pathlib import Path

import typer

JOB_NAME = "adv-opt-attempt2"


def construct_yaml() -> str:
    runner_template = (Path(__file__).parent / "runner.yaml").read_text()

    return runner_template.format(
        NAME=JOB_NAME,
        PRIORITY="normal-batch",
        COMMAND="ls -l $ACDC_OUTPUT_DIR",
        CONTAINER_TAG="latest",
        COMMIT_HASH="main",
        CPU="1",
        MEMORY="1Gi",
        GPU="0",
    )


# create a Typer app
app = typer.Typer()


@app.command()
def print():
    typer.echo(construct_yaml())


@app.command()
def launch():
    subprocess.run(["kubectl", "create", "-f", "-"], check=True, input=construct_yaml().encode())

    typer.echo("Job launched. Starting to follow logs...")
    subprocess.run(
        f"""
    until kubectl logs -f job/{JOB_NAME}; do
        echo "Retrying in 5 seconds..."
        sleep 5
    done
    """,
        shell=True,
        check=True,
    )


# run the app
if __name__ == "__main__":
    app()
