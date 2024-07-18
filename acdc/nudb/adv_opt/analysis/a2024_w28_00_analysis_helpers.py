from pathlib import Path

import jsonpickle
import torch
from transformers import AutoTokenizer

from acdc.nudb.adv_opt.analysis.analyzer_brute_force_v1 import (
    BruteForceExperimentOutputAnalysisV1,
    analyze_circuit_loss_metrics,
)
from acdc.nudb.adv_opt.brute_force.results import BruteForceResults
from acdc.nudb.adv_opt.data_fetchers import AdvOptTaskName
from acdc.nudb.adv_opt.masked_runner import MaskedRunner


def analyze_circuit_loss_distribution(
    result: BruteForceResults, tokenizer: AutoTokenizer, masked_runner: MaskedRunner
) -> BruteForceExperimentOutputAnalysisV1:
    return analyze_circuit_loss_metrics(
        tokenizer,
        input_tokens=torch.tensor(result.input),
        patch_input_tokens=torch.tensor(result.patch_input),
        metrics=torch.tensor(result.circuit_loss),
        circuit=result.circuit_spec.edges,
        masked_runner=masked_runner,
    )


CIRCUIT_LOSS_RESULT_PATHS: dict[AdvOptTaskName, Path] = {
    AdvOptTaskName.GREATERTHAN: Path(
        "/home/niels/data/advopt/raw/sync-2024-07-09T10_34_44/2024-07-08-234853_bruteforce_greaterthan/results_circuittype.canonical.json"
    ),
    AdvOptTaskName.DOCSTRING: Path(
        "/home/niels/data/advopt/raw/sync-2024-07-09T10_34_44/2024-07-08-232329_bruteforce_docstring/results_circuittype.canonical.json"
    ),
    AdvOptTaskName.IOI: Path(
        "/home/niels/data/advopt/raw/sync-2024-07-09T10_34_44/2024-07-08-223136_bruteforce_ioi/results_circuittype.canonical.json"
    ),
}


def load_circuit_loss_results(task_name: AdvOptTaskName) -> BruteForceResults:
    return jsonpickle.decode(CIRCUIT_LOSS_RESULT_PATHS[task_name].read_text())
