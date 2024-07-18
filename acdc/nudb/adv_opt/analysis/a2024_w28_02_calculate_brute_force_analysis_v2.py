from datetime import date
from pathlib import Path

import jsonpickle

from acdc.nudb.adv_opt.analysis.a2024_w28_00_analysis_helpers import (
    CIRCUIT_LOSS_RESULT_PATHS,
    analyze_circuit_loss_distribution,
    load_circuit_loss_results,
)
from acdc.nudb.adv_opt.analysis.analyzer_brute_force_v1 import (
    BruteForceExperimentOutputAnalysisV1,
)
from acdc.nudb.adv_opt.brute_force.results import BruteForceResults
from acdc.nudb.adv_opt.data_fetchers import AdvOptTaskName, get_standard_experiment_data


def calculate_analysis_from_raw_data(
    task_name: AdvOptTaskName, results: BruteForceResults
) -> BruteForceExperimentOutputAnalysisV1:
    experiment_data = get_standard_experiment_data(
        task_name, num_examples=10
    )  # we only do this because this is the easiest way to get the tokenizer and the masked runner...

    return analyze_circuit_loss_distribution(
        results, tokenizer=experiment_data.tokenizer, masked_runner=experiment_data.masked_runner
    )


if __name__ == "__main__":
    output_dir = Path(
        f"/home/niels/data/advopt/processed/{date.today()}-brute-force-loss-and-output-analysis-matched-corruptions"
    )
    print("Output dir:", output_dir)

    for task in CIRCUIT_LOSS_RESULT_PATHS:
        results = load_circuit_loss_results(task)
        analysis = calculate_analysis_from_raw_data(task, results)

        output_file = output_dir / str(task.value) / "analysis.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(jsonpickle.encode(analysis))

        print("Stored analysis for", task, "in", output_file)
