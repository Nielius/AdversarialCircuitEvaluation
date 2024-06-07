import subprocess
import warnings
from pathlib import Path
from typing import Any

from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from hydra.experimental.callback import Callback
from omegaconf import DictConfig


class GitSHACallback(Callback):
    def on_job_start(self, config: DictConfig, **kwargs: Any) -> None:
        hydra_folder = HydraConfig.get().runtime.output_dir
        commit_sha_file = Path(hydra_folder) / "sha.txt"
        with open(commit_sha_file, "w") as f:
            try:
                f.write(subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8"))
            except subprocess.CalledProcessError:
                warnings.warn("Could not get git commit sha.")


def load_git_sha_callback():
    ConfigStore.instance().store(
        name="git_sha_callback",
        node={
            "callbacks": {
                "git_sha": {
                    "_target_": GitSHACallback.__module__ + "." + GitSHACallback.__name__,
                }
            }
        },
        package="hydra",
    )
