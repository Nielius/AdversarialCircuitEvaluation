"""Load circuit loss distributions from disk and calculate the Wasserstein distance from such a distribution to
resamples of that distribution. The goal is to test whether or not we could have sampled fewer points and still
gotten good Wasserstein distance measurements."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import jsonpickle
import numpy as np
from scipy.stats import wasserstein_distance

from acdc.nudb.adv_opt.analysis.a2024_w18_analyze_different_prompts import IOIBruteForceResultsCollection


def calculate_resample_wasserstein_distances(
    circuit_loss: np.ndarray,
    n_samples: int,
    n_resamples: int,
) -> list[float]:
    """For a distribution of circuit losses, calculate the wasserstein distance between the original distribution and a
    number of smaller resamples."""
    return [
        wasserstein_distance(np.random.choice(circuit_loss, size=n_samples, replace=True), circuit_loss)
        for _ in range(n_resamples)
    ]


@dataclass
class ResampleDistance:
    result_index: int
    sample_size: int
    resample_distances: list[float]


if __name__ == "__main__":
    base_dir = Path("/home/niels/data/advopt/raw/tidy/2024-05-02-bruteforce-ioi-1000samples")
    results = IOIBruteForceResultsCollection.from_dir(base_dir)

    n_resamples = 100
    # sample_sizes = [10, 100]
    sample_sizes = [10, 100, 1000, 10000, 100000]
    all_distances: list[ResampleDistance] = []
    for result_index, result in enumerate(results.results):
        print("Starting with result index ", result_index)
        for sample_size in sample_sizes:
            start_time = datetime.now()
            print("Starting with sample size ", sample_size)
            all_distances.append(
                ResampleDistance(
                    result_index=result_index,
                    sample_size=sample_size,
                    resample_distances=calculate_resample_wasserstein_distances(
                        result.circuit_loss, n_samples=sample_size, n_resamples=n_resamples
                    ),
                )
            )
            end_time = datetime.now()
            print("Finished with sample size ", sample_size, " in ", end_time - start_time)

    Path(f"/home/niels/output-{datetime.now().isoformat()}.json").write_text(jsonpickle.dumps(all_distances))
