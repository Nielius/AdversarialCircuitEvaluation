import logging
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml
from jaxtyping import Float, Integer
from matplotlib import pyplot as plt
from transformers import AutoTokenizer
from typing_extensions import Self

from acdc.nudb.adv_opt.brute_force.results import CircuitPerformanceDistributionResultsV1
from acdc.nudb.adv_opt.data_fetchers import AdvOptExperimentData, AdvOptTaskName, get_standard_experiment_data
from acdc.nudb.adv_opt.masked_runner import MaskedRunner
from acdc.nudb.adv_opt.utils import DeepTokenDecoder
from acdc.TLACDCEdge import Edge

logger = logging.getLogger(__name__)


@dataclass
class BruteForceExperimentOutputAnalysisV1:
    """Analyzes the output of the circuit and the model on the strongest counterexamples found through brute force."""

    @dataclass
    class InputOutputData:
        input: str
        patch_input: str
        loss: float
        most_likely_output_circuit: list[str]
        most_likely_output_model: list[str]
        logits_circuit: list[float]
        logits_model: list[float]

    top_k_worst_inputs: list[InputOutputData]

    @classmethod
    def from_columns(
        cls,
        decoder: DeepTokenDecoder,
        losses: Float[torch.Tensor, " batch"],
        topk_most_adversarial_input: Integer[torch.Tensor, "batch pos"],
        topk_most_adversarial_patch_input: Integer[torch.Tensor, "batch pos"],
        top_k_most_likely_output_circuit: torch.return_types.topk,  # "batch output_idx"; indices are token indices; values are logits
        top_k_most_likely_output_model: torch.return_types.topk,
    ) -> Self:
        return cls(
            top_k_worst_inputs=[
                cls.InputOutputData(
                    input=decoder.decode(topk_most_adversarial_input[i]),
                    patch_input=decoder.decode(topk_most_adversarial_patch_input[i]),
                    loss=losses[i].item(),
                    most_likely_output_circuit=decoder.decode_individual_tokens(
                        top_k_most_likely_output_circuit.indices[i]
                    ),
                    most_likely_output_model=decoder.decode_individual_tokens(
                        top_k_most_likely_output_model.indices[i]
                    ),
                    logits_circuit=top_k_most_likely_output_circuit.values[i].tolist(),
                    logits_model=top_k_most_likely_output_model.values[i].tolist(),
                )
                for i in range(len(losses))
            ]
        )


@dataclass
class BruteForceExperimentAnalyzerV1:
    """Analyzes the v1 output data. This is the brute force that I had around the end of the MATS project."""

    results: CircuitPerformanceDistributionResultsV1
    config: dict

    @classmethod
    def from_dir(cls, experiment_dir: Path, experiment_name: AdvOptTaskName) -> "BruteForceExperimentAnalyzerV1":
        return cls(
            results=CircuitPerformanceDistributionResultsV1.load(
                experiment_dir / "artifacts", experiment_name=experiment_name, append_exp_name_to_dir=False
            ),
            config=yaml.safe_load((experiment_dir / ".hydra" / "config.yaml").read_text()),
        )

    def plot(self, output_dir: Path) -> None:
        num_circuits = len(self.results.metrics)
        fig, axes = plt.subplots(1, num_circuits, sharex=True, sharey=True)  # Create a figure containing a single axes.

        if num_circuits == 1:
            axes = [axes]

        # fig.suptitle(
        #     f"KL divergence between output of the full model and output of a circuit, for {self.results.experiment_name}, histogram"
        # )

        range = (
            0,
            max(metrics.max().item() for metrics in self.results.metrics.values()),
        )

        # If you want to do all the plots in one figure, you can do something like this:
        # ((ax_all, ax_1), (ax_2, ax_3)) = axes
        # ax_all.stairs(*torch.histogram(metrics_with_canonical_circuit, bins=100, range=range), label="canonical circuit")
        # ax_all.stairs(*torch.histogram(metrics_with_random_circuit, bins=100, range=range), label="random circuit")
        # ax_all.stairs(
        #     *torch.histogram(metrics_with_corrupted_canonical_circuit, bins=100, range=range),
        #     label="corrupted canonical circuit",
        # )
        for i, (circuit, metrics) in enumerate(self.results.metrics.items()):
            ax = axes[i]
            ax.stairs(*torch.histogram(metrics.to("cpu"), bins=100, range=range), label=circuit)

        for ax in axes:
            ax.set_xlabel("KL divergence")
            ax.set_ylabel("Frequency")
            ax.legend()

        plt.show()
        fig.savefig(figure_path := output_dir / "histogram.svg")
        fig.savefig(output_dir / "histogram.pdf")
        logger.info("Saved histogram to %s", figure_path)

    def get_experiment_data(self) -> AdvOptExperimentData:
        return get_standard_experiment_data(AdvOptTaskName[self.config["task"]["task_name"]])

    def analyse_all_metrics(
        self, experiment_data: AdvOptExperimentData | None = None
    ) -> list[BruteForceExperimentOutputAnalysisV1]:
        experiment_data = experiment_data or self.get_experiment_data()
        results = []
        for circuit_name, metric in self.results.metrics.items():
            match circuit_name:
                case "canonical":
                    circuit = experiment_data.circuit_edges
                case _:
                    raise NotImplementedError()
            results.append(self.analyse_metrics(metric, circuit, experiment_data))

        return results

    def analyse_metrics(
        self, metrics: Float[torch.Tensor, " batch"], circuit: list[Edge], experiment_data: AdvOptExperimentData
    ) -> BruteForceExperimentOutputAnalysisV1:
        return analyze_circuit_loss_metrics(
            experiment_data.masked_runner.masked_transformer.model.tokenizer,
            self.results.test_data,
            self.results.test_patch_data,
            metrics,
            circuit,
            experiment_data.masked_runner,
        )

    def analyse_and_print_metrics(
        self, metrics: Float[torch.Tensor, " batch"], circuit: list[Edge], experiment_data: AdvOptExperimentData
    ) -> None:
        print_circuit_loss_metrics(self.analyse_metrics(metrics, circuit, experiment_data))

    def analyse_and_print_all_metrics(self, experiment_data: AdvOptExperimentData | None = None) -> None:
        analyses = self.analyse_all_metrics(experiment_data)
        for analysis in analyses:
            print_circuit_loss_metrics(analysis)


def analyze_circuit_loss_metrics(
    tokenizer: AutoTokenizer,
    input_tokens: Integer[torch.Tensor, "batch pos"],
    patch_input_tokens: Integer[torch.Tensor, "batch pos"],
    metrics: Float[torch.Tensor, " batch"],
    circuit: list[Edge],
    masked_runner: MaskedRunner,
) -> BruteForceExperimentOutputAnalysisV1:
    """Analyze the metrics of the circuit loss, meaning that we provide:

    - for the top_k_input worst inputs,
      - we show the top_k_out most likely outputs from the circuit and the model
      - the loss obtained by those inputs
    """

    top_k_input = 10
    top_k_out = 3  # how many of the most likely output to show

    topk_most_adversarial = torch.topk(metrics, k=top_k_input, sorted=True)
    topk_most_adversarial_input = input_tokens[topk_most_adversarial.indices, :]
    topk_most_adversarial_patch_input = patch_input_tokens[topk_most_adversarial.indices, :]

    topk_losses = topk_most_adversarial.values

    # output of circuit and output of full model
    output_circuit = masked_runner.run(
        input=topk_most_adversarial_input,
        patch_input=topk_most_adversarial_patch_input,
        edges_to_ablate=list(masked_runner.all_ablatable_edges - set(circuit)),
    )
    top_k_most_likely_output_circuit = torch.topk(output_circuit[:, -1, :], k=top_k_out)

    output_model = masked_runner.run(
        input=topk_most_adversarial_input,
        patch_input=topk_most_adversarial_patch_input,
        edges_to_ablate=[],
    )
    top_k_most_likely_output_model = torch.topk(output_model[:, -1, :], k=top_k_out)

    return BruteForceExperimentOutputAnalysisV1.from_columns(
        decoder=DeepTokenDecoder(tokenizer),
        losses=topk_losses,
        topk_most_adversarial_input=topk_most_adversarial_input,
        topk_most_adversarial_patch_input=topk_most_adversarial_patch_input,
        top_k_most_likely_output_circuit=top_k_most_likely_output_circuit,
        top_k_most_likely_output_model=top_k_most_likely_output_model,
    )


def print_circuit_loss_metrics(analysis: BruteForceExperimentOutputAnalysisV1) -> None:
    for input_output in analysis.top_k_worst_inputs:
        logger.info("------------------------------------")
        logger.info("Input: %s", input_output.input)
        logger.info("Patch input: %s", input_output.patch_input)
        logger.info("Loss: %s", input_output.loss)
        logger.info("Most likely output from circuit: %s", input_output.most_likely_output_circuit)
        logger.info("Most likely output from model: %s", input_output.most_likely_output_model)


def analyze_and_print_circuit_loss_metrics(
    tokenizer: AutoTokenizer,
    input_tokens: Integer[torch.Tensor, "batch pos"],
    patch_input_tokens: Integer[torch.Tensor, "batch pos"],
    metrics: Float[torch.Tensor, " batch"],
    circuit: list[Edge],
    masked_runner: MaskedRunner,
) -> None:
    """Analyze the metrics of the circuit loss, meaning that we provide:

    - for the top_k_input worst inputs,
      - we show the top_k_out most likely outputs from the circuit and the model
      - the loss
    """

    print_circuit_loss_metrics(
        analyze_circuit_loss_metrics(tokenizer, input_tokens, patch_input_tokens, metrics, circuit, masked_runner)
    )
