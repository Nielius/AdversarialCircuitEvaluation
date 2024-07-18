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
import pandas as pd
import seaborn as sns
import torch

from acdc.ioi.ioi_dataset_v2 import IOI_PROMPT_TEMPLATE_TOKEN_POSITIONS
from acdc.nudb.adv_opt.brute_force.results import (
    BruteForceResults,
    CircuitPerformanceDistributionResultsV1,
    IOIBruteForceResults,
)
from acdc.nudb.adv_opt.data_fetchers import AdvOptTaskName, get_standard_experiment_data
from acdc.nudb.adv_opt.utils import DeepTokenDecoder

# %%
# Conclusion: token 12 is the location, token 17 is the object

DEFAULT_PERCENTILES = [0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999, 0.9999]


class GroupedLossDistributionAnalyzer:
    _results: BruteForceResults
    _token_decoder: DeepTokenDecoder
    _loss_df: pd.DataFrame

    def __init__(self, results: BruteForceResults, token_decoder: DeepTokenDecoder):
        self._results = results
        self._token_decoder = token_decoder
        self._loss_df = self.calculate_grouped_distribution_description(results)

    @staticmethod
    def calculate_grouped_distribution_description(results: BruteForceResults) -> pd.DataFrame:
        # dataframe with the losses + object/location
        assert isinstance(results, IOIBruteForceResults)
        object_token_position = IOI_PROMPT_TEMPLATE_TOKEN_POSITIONS[results.prompt_template_index]["object"]
        location_token_position = IOI_PROMPT_TEMPLATE_TOKEN_POSITIONS[results.prompt_template_index]["place"]

        return (
            pd.DataFrame(results.circuit_loss, columns=["loss"])
            .assign(object=pd.Series(results.input[:, object_token_position]))
            .assign(location=pd.Series(results.input[:, location_token_position]))
            .assign(patch_object=pd.Series(results.patch_input[:, object_token_position]))
            .assign(patch_location=pd.Series(results.patch_input[:, location_token_position]))
        )

    def plot_heatmap(self, pivot_args: tuple[str, str, str], fig=None, ax=None):
        """This is grouped by 2 fields"""
        if (fig is None) != (ax is None):
            raise ValueError("Both fig and ax must be provided or neither")

        first, second, value = pivot_args
        heatmap_data = (
            self._loss_df[[first, second, "loss"]]
            .groupby([first, second])
            .describe(percentiles=DEFAULT_PERCENTILES)
            .reset_index()
            .pivot(index=first, columns=second, values=value)
        )

        heatmap_data.index = heatmap_data.index.map(lambda x: self._token_decoder.decode(torch.tensor(x)))
        heatmap_data.columns = heatmap_data.columns.map(lambda x: self._token_decoder.decode(torch.tensor(x)))

        heatmap_data.sort_index(axis=0, inplace=True)
        heatmap_data.sort_index(axis=1, inplace=True)

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))

        # Create the heatmap
        sns.heatmap(heatmap_data, cmap="YlGnBu", annot=True, fmt=".2f", ax=ax)
        ax.set_title(f"Heatmap of {value} by {first} and {second}")
        ax.set_xlabel(second)
        ax.set_ylabel(first)

        if ax is None:
            plt.show()

    def calculate_distribution_description(self, column_name: str):
        """This is grouped by only a single field"""
        description_df = (
            self._loss_df[[column_name, "loss"]].groupby([column_name]).describe(percentiles=DEFAULT_PERCENTILES)
        )
        description_df.index = description_df.index.map(lambda x: self._token_decoder.decode(torch.tensor(x)))
        description_df.sort_index(inplace=True)
        return description_df


# %%

tokenizer = get_standard_experiment_data(
    AdvOptTaskName.IOI, num_examples=10
).tokenizer  # we only do this because this is the easiest way to get the tokenizer and the masked runner...

results_with_paired_patches: BruteForceResults = jsonpickle.decode(
    Path(
        "/home/niels/data/advopt/raw/tidy/2024-07-10-bruteforce-1M-samples-matched-corruptions/2024-07-09-094807_bruteforce_ioi/results_circuittype.canonical.json"
    ).read_text()
)

# %%

analyzer = GroupedLossDistributionAnalyzer(results_with_paired_patches, DeepTokenDecoder(tokenizer))

# analyzer.calculate_distribution_description("location")
# analyzer.plot_heatmap(("location", "object", ("loss", "max")))

fig, axes = plt.subplots(1, 3, figsize=(20, 8))

analyzer.plot_heatmap(("location", "object", ("loss", "max")), fig=fig, ax=axes[0])
analyzer.plot_heatmap(("location", "object", ("loss", "99.9%")), fig=fig, ax=axes[1])
analyzer.plot_heatmap(("location", "object", ("loss", "99.99%")), fig=fig, ax=axes[2])

fig.savefig("/home/niels/heatmap-matched.pdf")


# %%

# /home/niels/data/advopt/raw/tidy/2024-03-02-bruteforce-v1
# ./2024-03-02-130221_bruteforce_docstring_1/artifacts
# ./2024-03-02-081117_bruteforce_greaterthan_1/artifacts
# ./2024-03-02-011541_bruteforce_ioi_1/artifacts

results_with_arbitrary_patches: BruteForceResults = CircuitPerformanceDistributionResultsV1.load(
    Path("/home/niels/data/advopt/raw/tidy/2024-03-02-bruteforce-v1/2024-03-02-011541_bruteforce_ioi_1/artifacts"),
    AdvOptTaskName.IOI,
    append_exp_name_to_dir=False,
).convert_to_brute_force_results()

analyzer = GroupedLossDistributionAnalyzer(results_with_arbitrary_patches, DeepTokenDecoder(tokenizer))

# %%

fig, axes = plt.subplots(1, 3, figsize=(20, 8))

analyzer.plot_heatmap(("location", "object", ("loss", "max")), fig=fig, ax=axes[0])
analyzer.plot_heatmap(("location", "object", ("loss", "99.9%")), fig=fig, ax=axes[1])
analyzer.plot_heatmap(("location", "object", ("loss", "99.99%")), fig=fig, ax=axes[2])

fig.savefig("/home/niels/heatmap.pdf")

# %%
fig, axes = plt.subplots(1, 3, figsize=(20, 8))

analyzer.plot_heatmap(("patch_object", "object", ("loss", "max")), fig=fig, ax=axes[0])
analyzer.plot_heatmap(("patch_object", "object", ("loss", "99.9%")), fig=fig, ax=axes[1])
analyzer.plot_heatmap(("patch_object", "object", ("loss", "99.99%")), fig=fig, ax=axes[2])

fig.savefig("/home/niels/heatmap-patch-object.pdf")


# %%
fig, axes = plt.subplots(1, 3, figsize=(20, 8))

analyzer.plot_heatmap(("patch_location", "location", ("loss", "max")), fig=fig, ax=axes[0])
analyzer.plot_heatmap(("patch_location", "location", ("loss", "99.9%")), fig=fig, ax=axes[1])
analyzer.plot_heatmap(("patch_location", "location", ("loss", "99.99%")), fig=fig, ax=axes[2])

fig.savefig("/home/niels/heatmap-patch-location.pdf")

# %%
