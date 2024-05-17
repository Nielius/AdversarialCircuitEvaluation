from pathlib import Path

import jsonpickle
import matplotlib.pyplot as plt
import pandas as pd

from acdc.nudb.adv_opt.analysis.a2024_w20_01_calculate_wasserstein_resample_distances import ResampleDistance

if __name__ == "__main__":
    result: list[ResampleDistance] = jsonpickle.decode(
        Path(
            "/home/niels/data/advopt/raw/tidy/2024-05-13-wasserstein-distances-from-resamples/wasserstein-distances-from-resamples.json"
        ).read_text()
    )
    distances_df = pd.DataFrame(result).set_index(["result_index", "sample_size"]).explode("resample_distances")

    # Calculate boxplot for all the resample distances
    axes: plt.Axes = distances_df.reset_index().boxplot(column="resample_distances", by="sample_size")
    axes.set_ylabel("Wasserstein distance from base distributions")
    axes.set_yscale("log")
    axes.legend()
    axes.figure.savefig(
        "/home/niels/data/advopt/processed/2024-05-14-wasserstein-distances-resampled/boxplot-per-sample-size.svg"
    )
    plt.show()

    # Calculate boxplot for all the resample distances, but grouped by result_index
    # Helps us answer whether or not it differs per prompt
    axes = distances_df.boxplot(column="resample_distances", by=["sample_size", "result_index"])
    axes.set_ylabel("Wasserstein distance from base distributions")
    axes.set_yscale("log")
    axes.tick_params(axis="x", labelrotation=90)
    axes.figure.subplots_adjust(bottom=0.3)
    axes.figure.savefig(
        "/home/niels/data/advopt/processed/2024-05-14-wasserstein-distances-resampled/boxplot-per-sample-size-per-prompt.svg"
    )
    plt.show()
