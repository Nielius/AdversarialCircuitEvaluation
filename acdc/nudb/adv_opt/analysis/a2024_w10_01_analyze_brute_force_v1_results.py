"""Script to analyze the brute force results from 2024-03."""

import logging
from dataclasses import dataclass
from pathlib import Path

import hydra
import jsonpickle
from hydra.core import hydra_config
from hydra.core.config_store import ConfigStore
from icecream import ic

from acdc.nudb.adv_opt import hydra_utils
from acdc.nudb.adv_opt.analysis.analyzer_brute_force_v1 import (
    BruteForceExperimentAnalyzerV1,
    print_circuit_loss_metrics,
)
from acdc.nudb.adv_opt.data_fetchers import AdvOptTaskName
from acdc.nudb.adv_opt.utils import ADVOPT_DATA_DIR


@dataclass
class AnalysisSettings:
    task_name: AdvOptTaskName = AdvOptTaskName.GREATERTHAN
    input_dir: Path | None = None


def get_default_input_dir(task_name: AdvOptTaskName) -> Path:
    raw_outputs_base_dir = ADVOPT_DATA_DIR / "raw/tidy/2024-03-02-bruteforce-v1"
    experiment_path: dict[AdvOptTaskName, Path] = {
        AdvOptTaskName.IOI: raw_outputs_base_dir / "2024-03-02-011541_bruteforce_ioi_1",
        AdvOptTaskName.GREATERTHAN: raw_outputs_base_dir / "2024-03-02-081117_bruteforce_greaterthan_1",
        AdvOptTaskName.TRACR_REVERSE: raw_outputs_base_dir / "2024-03-02-130207_bruteforce_tracr_reverse_1",
        AdvOptTaskName.DOCSTRING: raw_outputs_base_dir / "2024-03-02-130221_bruteforce_docstring_1",
    }
    return experiment_path[task_name]


cs = ConfigStore.instance()
cs.store(name="settings_schema", node=AnalysisSettings)
hydra_utils.load_git_sha_callback()

logger = logging.getLogger(__name__)
ic.configureOutput(outputFunction=logger.info)


@hydra.main(config_name="a2024_w10_01_analyze_brute_force_v1_results.yaml", version_base=None, config_path=".")
def main(settings: AnalysisSettings):
    task_name = settings.task_name
    output_base_dir = Path(hydra_config.HydraConfig.get().runtime.output_dir)
    input_dir = settings.input_dir or get_default_input_dir(task_name)

    experiment_analyzer = BruteForceExperimentAnalyzerV1.from_dir(input_dir, task_name)

    experiment_analyzer.plot(output_base_dir)

    circuit_output_analyses = experiment_analyzer.analyse_all_metrics()
    (output_base_dir / "analyses.json").write_text(jsonpickle.dumps(circuit_output_analyses))

    for analysis in circuit_output_analyses:
        print_circuit_loss_metrics(analysis)

    logger.info("Finished analysis of brute force results for task %s; results in %s", task_name, output_base_dir)


if __name__ == "__main__":
    main()
