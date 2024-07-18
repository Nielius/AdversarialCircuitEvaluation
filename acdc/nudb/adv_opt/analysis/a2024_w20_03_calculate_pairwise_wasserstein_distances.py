"""Load circuit loss distributions from disk and calculate the Wasserstein distances between those distributions.
The question we want to answer is whether prompts that are similar enough for the same task lead to loss distributions
that are close in Wasserstein distance."""

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import typer
from scipy.stats import wasserstein_distance

from acdc.nudb.adv_opt.analysis.a2024_w18_analyze_different_prompts import IOIBruteForceResultsCollection


def calculate_pairwise_wasserstein_distances(results: IOIBruteForceResultsCollection) -> np.ndarray:
    """Calculate pairwise wasserstein distances between the circuit loss distributions of the different prompts."""
    result = np.zeros((len(results.dirs_and_results), len(results.dirs_and_results)))

    for (i, result1), (j, result2) in combinations(enumerate(results.results), r=2):
        distance = wasserstein_distance(result1.circuit_loss, result2.circuit_loss)
        result[i, j] = distance
        result[j, i] = distance

    return result


def main(results: IOIBruteForceResultsCollection, result_dir: Path):
    pairwise_distances = calculate_pairwise_wasserstein_distances(results)

    metadata_df = pd.DataFrame(
        [
            (
                result.prompt_template_index,
                "".join(dir.config.cfg_dict["task"]["names_order"]),
                result.prompt_template,
                dir.config.cfg_dict["task"]["task_name"],
                dir.config.cfg_dict["num_examples"],
            )
            for dir, result in results.dirs_and_results
        ],
        columns=["prompt_template_index", "names_order", "prompt_template", "task_name", "num_examples"],
    )

    pairwise_distances_df = pd.DataFrame(pairwise_distances, index=metadata_df.index, columns=metadata_df.index)

    pairwise_distances_df.to_csv(result_dir / "pairwise_distances.csv")
    metadata_df.to_csv(result_dir / "metadata.csv")


def main_from_paths(result_dir: Path, input_dirs: list[Path]):
    results: IOIBruteForceResultsCollection = IOIBruteForceResultsCollection.merge(
        *(IOIBruteForceResultsCollection.from_dir(input_dir) for input_dir in input_dirs)
    )
    main(results, result_dir)


if __name__ == "__main__":
    typer.run(main_from_paths)
