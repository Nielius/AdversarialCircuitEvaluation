"""Script to analyze the brute force results from 2024-03."""

import logging
from dataclasses import dataclass
from pathlib import Path

import hydra
from hydra.core import hydra_config
from hydra.core.config_store import ConfigStore
from icecream import ic

from acdc.nudb.adv_opt.analysis.analyzer_brute_force_old import BruteForceExperimentAnalysisOld
from acdc.nudb.adv_opt.data_fetchers import AdvOptTaskName

raw_outputs_base_dir = Path("/home/niels/proj/mats/data/outputs/brute-force-with-corrupted")

experiment_path: dict[AdvOptTaskName, Path] = {
    AdvOptTaskName.IOI: raw_outputs_base_dir / "2024-03-02-011541_bruteforce_ioi",
    AdvOptTaskName.GREATERTHAN: raw_outputs_base_dir / "2024-03-02-081117_bruteforce_greaterthan",
    AdvOptTaskName.TRACR_REVERSE: raw_outputs_base_dir / "2024-03-02-130207_bruteforce_tracr_reverse",
    AdvOptTaskName.DOCSTRING: raw_outputs_base_dir / "2024-03-02-130221_bruteforce_docstring",
}


@dataclass
class AnalysisSettings:
    task_name: AdvOptTaskName = AdvOptTaskName.GREATERTHAN


cs = ConfigStore.instance()
cs.store(name="settings_schema", node=AnalysisSettings)

logger = logging.getLogger(__name__)
ic.configureOutput(outputFunction=logger.info)


@hydra.main(config_name="analyze_brute_force_results.yaml")
def main(settings: AnalysisSettings):
    task_name = settings.task_name
    output_base_dir = Path(hydra_config.HydraConfig.get().runtime.output_dir)
    experiment_analysis = BruteForceExperimentAnalysisOld.from_dir(experiment_path[task_name], task_name)

    experiment_analysis.plot(output_base_dir)
    experiment_analysis.analyse_all_metrics()

    logger.info("Finished analysis of brute force results for task %s; results in %s", task_name, output_base_dir)


if __name__ == "__main__":
    main()
