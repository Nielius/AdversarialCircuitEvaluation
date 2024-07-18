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
        if results.task_name == AdvOptTaskName.IOI:
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

        elif results.task_name == AdvOptTaskName.GREATERTHAN:
            event_token_position = 1
            century_token_position = 6
            first_year_position = 7
            # {0: 'The',
            # 1: ' sanctions',
            # 2: ' lasted',
            # 3: ' from',
            # 4: ' the',
            # 5: ' year',
            # 6: ' 17',
            # 7: '24',
            # 8: ' to',
            # 9: ' 17'}

            return (
                pd.DataFrame(results.circuit_loss, columns=["loss"])
                .assign(event=pd.Series(results.input[:, event_token_position]))
                .assign(century=pd.Series(results.input[:, century_token_position]))
                .assign(year=pd.Series(results.input[:, first_year_position]))
                .assign(patch_event=pd.Series(results.patch_input[:, event_token_position]))
                .assign(patch_century=pd.Series(results.patch_input[:, century_token_position]))
                .assign(patch_year=pd.Series(results.patch_input[:, first_year_position]))
            )

        elif results.task_name == AdvOptTaskName.DOCSTRING:
            raise NotImplementedError("This might not make sense. There are too many of these?")

            #        {0: '<|BOS|>',
            # 1: '\n',
            # 2: 'def',
            # 3: ' request',
            # 4: '(',
            # 5: 'self',
            # 6: ',',
            # 7: ' function',
            # 8: ',',
            # 9: ' number',
            # 10: ',',
            # 11: ' expected',
            # 12: ',',
            # 13: ' results',
            # 14: ',',
            # 15: ' model',
            # 16: ',',
            # 17: ' order',
            # 18: '):',
            # 19: '\n   ',
            # 20: ' """',
            # 21: 'story',
            # 22: ' flight',
            # 23: ' data',
            # 24: '\n\n   ',
            # 25: ' :',
            # 26: 'param',
            # 27: ' expected',
            # 28: ':',
            # 29: ' notice',
            # 30: ' expression',
            # 31: '\n   ',
            # 32: ' :',
            # 33: 'param',
            # 34: ' results',
            # 35: ':',
            # 36: ' wind',
            # 37: ' cup',
            # 38: '\n   ',
            # 39: ' :',
            # 40: 'param'}

        else:
            raise ValueError(f"Unknown task name {results.task_name}")

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
        sns.heatmap(
            heatmap_data,
            cmap="YlGnBu",
            annot=True if self._results.task_name != AdvOptTaskName.GREATERTHAN else False,
            fmt=".2f",
            ax=ax,
        )
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
    AdvOptTaskName.GREATERTHAN, num_examples=10
).tokenizer  # we only do this because this is the easiest way to get the tokenizer and the masked runner...

results_with_paired_patches: BruteForceResults = jsonpickle.decode(
    Path(
        "/home/niels/data/advopt/raw/sync-2024-07-09T10_34_44/2024-07-08-150027_bruteforce_greaterthan/results_circuittype.canonical.json"
        # "/home/niels/data/advopt/raw/tidy/2024-07-10-bruteforce-1M-samples-matched-corruptions/2024-07-09-225415_bruteforce_greaterthan/results_circuittype.canonical.json"
        # "/home/niels/data/advopt/raw/tidy/2024-07-10-bruteforce-1M-samples-matched-corruptions/2024-07-09-182856_bruteforce_docstring/results_circuittype.canonical.json"
    ).read_text()
)

# %%
analyzer = GroupedLossDistributionAnalyzer(results_with_paired_patches, DeepTokenDecoder(tokenizer))

# analyzer.calculate_distribution_description("location")
# analyzer.plot_heatmap(("location", "object", ("loss", "max")))

fig, axes = plt.subplots(1, 3, figsize=(50, 16))

analyzer.plot_heatmap(("event", "year", ("loss", "max")), fig=fig, ax=axes[0])
analyzer.plot_heatmap(("event", "year", ("loss", "99.9%")), fig=fig, ax=axes[1])
analyzer.plot_heatmap(("event", "year", ("loss", "99.99%")), fig=fig, ax=axes[2])

fig.savefig("/home/niels/heatmap-greaterthan-matched.pdf")


# %%
# /home/niels/data/advopt/raw/tidy/2024-03-02-bruteforce-v1
# ./2024-03-02-130221_bruteforce_docstring_1/artifacts
# ./2024-03-02-081117_bruteforce_greaterthan_1/artifacts
# ./2024-03-02-011541_bruteforce_ioi_1/artifacts

results_with_arbitrary_patches: BruteForceResults = CircuitPerformanceDistributionResultsV1.load(
    Path(
        "/home/niels/data/advopt/raw/tidy/2024-03-02-bruteforce-v1/2024-03-02-081117_bruteforce_greaterthan_1/artifacts"
    ),
    AdvOptTaskName.GREATERTHAN,
    append_exp_name_to_dir=False,
).convert_to_brute_force_results()

analyzer = GroupedLossDistributionAnalyzer(results_with_arbitrary_patches, DeepTokenDecoder(tokenizer))

# %%
fig, axes = plt.subplots(1, 3, figsize=(20, 8))

analyzer.plot_heatmap(("event", "year", ("loss", "max")), fig=fig, ax=axes[0])
analyzer.plot_heatmap(("event", "year", ("loss", "99.9%")), fig=fig, ax=axes[1])
analyzer.plot_heatmap(("event", "year", ("loss", "99.99%")), fig=fig, ax=axes[2])

fig.savefig("/home/niels/heatmap-greaterthan.pdf")

# %%
fig, axes = plt.subplots(1, 3, figsize=(20, 8))

analyzer.plot_heatmap(("patch_year", "year", ("loss", "max")), fig=fig, ax=axes[0])
analyzer.plot_heatmap(("patch_year", "year", ("loss", "99.9%")), fig=fig, ax=axes[1])
analyzer.plot_heatmap(("patch_year", "year", ("loss", "99.99%")), fig=fig, ax=axes[2])

fig.savefig("/home/niels/heatmap-greaterthan-patch-year.pdf")


# %%
fig, axes = plt.subplots(1, 3, figsize=(20, 8))

analyzer.plot_heatmap(("patch_event", "event", ("loss", "max")), fig=fig, ax=axes[0])
analyzer.plot_heatmap(("patch_event", "event", ("loss", "99.9%")), fig=fig, ax=axes[1])
analyzer.plot_heatmap(("patch_event", "event", ("loss", "99.99%")), fig=fig, ax=axes[2])

fig.savefig("/home/niels/heatmap-greaterthan-patch-location.pdf")

# %%
