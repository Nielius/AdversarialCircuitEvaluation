# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
from pathlib import Path

import jsonpickle
import matplotlib.pyplot as plt
import torch

from acdc.nudb.adv_opt.brute_force.results import BruteForceResults

# %%
output_dir = Path("/home/niels/tmp")


# %%
def load_and_plot_distribution(input_dir: Path, output_base_name: Path | None = None):
    results: BruteForceResults = jsonpickle.decode(input_dir.read_text())

    metrics = results.circuit_loss
    fig, ax = plt.subplots()

    data_range = (0, metrics.max())

    ax.set_xlabel("KL divergence")
    ax.set_ylabel("Frequency")

    ax.stairs(*torch.histogram(torch.tensor(metrics), bins=100, range=data_range))

    if output_base_name is not None:
        fig.savefig(output_base_name.with_suffix(".svg"))
        fig.savefig(output_base_name.with_suffix(".pdf"))

    plt.show()


# %%
# Greaterthan
load_and_plot_distribution(
    Path(
        "/home/niels/data/advopt/raw/sync-2024-07-09T10_34_44/2024-07-08-234853_bruteforce_greaterthan/results_circuittype.canonical.json"
    ),
    output_base_name=output_dir / "greaterthan-histogram-matched",
)

# %%
# Docstring
load_and_plot_distribution(
    Path(
        "/home/niels/data/advopt/raw/sync-2024-07-09T10_34_44/2024-07-08-232329_bruteforce_docstring/results_circuittype.canonical.json"
    ),
    output_base_name=output_dir / "docstring-histogram-matched",
)

# %%
# IOI
print("IOI")
load_and_plot_distribution(
    Path(
        "/home/niels/data/advopt/raw/sync-2024-07-09T10_34_44/2024-07-08-223136_bruteforce_ioi/results_circuittype.canonical.json"
    ),
    output_base_name=output_dir / "ioi-histogram-matched",
)
