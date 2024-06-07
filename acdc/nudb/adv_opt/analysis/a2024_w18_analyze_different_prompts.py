"""Script to analyze the brute force results for the IOI task with the new IOI dataset v2.

The output data that is being analyzed is encoded differently than the data from earlier brute force experiments
(before April 2024).
"""

import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import pandas as pd
import torch
from transformers import AutoTokenizer
from typing_extensions import Self

from acdc.ioi.ioi_dataset_v2 import get_ioi_tokenizer
from acdc.nudb.adv_opt.analysis.analyzer_brute_force_v1 import analyze_and_print_circuit_loss_metrics
from acdc.nudb.adv_opt.analysis.output_parser import AdvOptBruteForceOutputDir
from acdc.nudb.adv_opt.brute_force.circuit_edge_fetcher import CircuitType
from acdc.nudb.adv_opt.brute_force.results import (
    IOIBruteForceResults,
)
from acdc.nudb.adv_opt.data_fetchers import get_adv_opt_data_provider_for_ioi
from acdc.nudb.adv_opt.masked_runner import MaskedRunner

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


def analyze_worst_inputs_with_outputs(
    masked_runner: MaskedRunner, tokenizer: AutoTokenizer, result: IOIBruteForceResults
):
    input_tokens = torch.tensor(result.input)
    patch_input_tokens = torch.tensor(result.patch_input)
    circuit_loss = torch.tensor(result.circuit_loss)

    analyze_and_print_circuit_loss_metrics(
        tokenizer,
        input_tokens,
        patch_input_tokens,
        circuit_loss,
        result.circuit_spec.edges,
        masked_runner,
    )


@dataclass
class IOIBruteForceResultsCollection:
    dirs_and_results: list[tuple[AdvOptBruteForceOutputDir, IOIBruteForceResults]]

    @property
    def results(self):
        return [result for _, result in self.dirs_and_results]

    @property
    def dirs(self):
        return [dir for dir, _ in self.dirs_and_results]

    @classmethod
    def from_dir(cls, base_dir: Path) -> Self:
        all_results = [
            (
                experiment_dir := AdvOptBruteForceOutputDir(experiment_dir_path),
                cast(IOIBruteForceResults, experiment_dir.result(CircuitType.CANONICAL)),
            )
            for experiment_dir_path in sorted(list(base_dir.glob("*/")))  # sorted for determinism
            if experiment_dir_path.is_dir()
        ]
        return cls(dirs_and_results=all_results)

    def print_results(self):
        # Print all results
        for experiment_dir, result in self.dirs_and_results:
            names_order = "".join(experiment_dir.config.cfg_dict["task"]["names_order"])
            print(
                f"Circuit loss statistics for {names_order}, prompt {result.prompt_template_index}: {result.prompt_template}",
            )
            print(pd.DataFrame(result.circuit_loss).describe())

    def print_statistics(self):
        results = self.dirs_and_results
        df_metadata = pd.DataFrame(
            {
                "prompt_template_index": [result.prompt_template_index for _, result in results],
                "names_order": [
                    "".join(experiment_dir.config.cfg_dict["task"]["names_order"]) for experiment_dir, _ in results
                ],
                "prompt_template": [result.prompt_template for _, result in results],
            }
        )

        df_loss_only = pd.DataFrame(
            {
                (df_metadata.loc[i, "prompt_template_index"], df_metadata.loc[i, "names_order"]): result.circuit_loss
                for i, (_, result) in enumerate(results)
            }
        )

        print(df_metadata)

        df_loss_only_statistics = df_loss_only.describe()
        print(df_loss_only_statistics)
        print(df_loss_only_statistics.loc["max"].describe())
        print(df_loss_only_statistics.loc["mean"].describe())

        df_loss_only.plot.hist(bins=100, histtype="step")  # this plots all in one plot
        df_loss_only.hist(bins=100, sharex=True, sharey=True)  # this plots in separate plots
        plt.show()


if __name__ == "__main__":
    # base_dir = Path("/home/niels/data/advopt/raw/tidy/2024-05-01-bruteforce-ioi-500samples")
    base_dir = Path("/home/niels/data/advopt/raw/tidy/2024-05-02-bruteforce-ioi-1000samples")

    results = IOIBruteForceResultsCollection.from_dir(base_dir)

    results.print_results()
    results.print_statistics()

    tokenizer = get_ioi_tokenizer()
    masked_runner = (
        get_adv_opt_data_provider_for_ioi(random.Random(0), template_index=0, names_order=list("ABBA"))
        .get_experiment_data(num_examples=2, metric_name="kl_div", device="cpu")
        .masked_runner
    )
    for _, result in results.dirs_and_results:
        analyze_worst_inputs_with_outputs(masked_runner, tokenizer, result)


def plot(circuit_loss, results):
    fig, axes = plt.subplots(1, 1, sharex=True, sharey=True)  # Create a figure containing a single axes.

    if True:  # num_circuits == 1:
        axes = [axes]

    fig.suptitle("KL divergence between output of the full model and output of a circuit")

    axes.stairs(*torch.histogram(circuit_loss, bins=100), label="canonical circuit")
    plt.show()

    range = (0, circuit_loss.max().item())

    # If you want to do all the plots in one figure, you can do something like this:
    # ((ax_all, ax_1), (ax_2, ax_3)) = axes
    # ax_all.stairs(*torch.histogram(metrics_with_canonical_circuit, bins=100, range=range), label="canonical circuit")
    # ax_all.stairs(*torch.histogram(metrics_with_random_circuit, bins=100, range=range), label="random circuit")
    # ax_all.stairs(
    #     *torch.histogram(metrics_with_corrupted_canonical_circuit, bins=100, range=range),
    #     label="corrupted canonical circuit",
    # )
    for i, (circuit, metrics) in enumerate(results.metrics.items()):
        ax = axes[i]
        ax.stairs(*torch.histogram(metrics, bins=100, range=range), label=circuit)

    for ax in axes:
        ax.set_xlabel("KL divergence")
        ax.set_ylabel("Frequency")
        ax.legend()

    plt.show()
