import logging
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml
from jaxtyping import Float, Integer
from matplotlib import pyplot as plt
from transformers import AutoTokenizer

from acdc.nudb.adv_opt.brute_force.results import CircuitPerformanceDistributionResultsOld
from acdc.nudb.adv_opt.data_fetchers import AdvOptExperimentData, AdvOptTaskName, get_standard_experiment_data
from acdc.nudb.adv_opt.masked_runner import MaskedRunner
from acdc.nudb.adv_opt.utils import deep_map_with_depth
from acdc.TLACDCEdge import Edge

logger = logging.getLogger(__name__)


@dataclass
class BruteForceExperimentAnalysisOld:
    results: CircuitPerformanceDistributionResultsOld
    config: dict

    @classmethod
    def from_dir(cls, experiment_dir: Path, experiment_name: AdvOptTaskName) -> "BruteForceExperimentAnalysisOld":
        return cls(
            results=CircuitPerformanceDistributionResultsOld.load(
                experiment_dir / "artifacts", experiment_name=experiment_name, append_exp_name_to_dir=False
            ),
            config=yaml.safe_load((experiment_dir / ".hydra" / "config.yaml").read_text()),
        )

    def plot(self, output_dir: Path) -> None:
        num_circuits = len(self.results.metrics)
        fig, axes = plt.subplots(1, num_circuits, sharex=True, sharey=True)  # Create a figure containing a single axes.

        if num_circuits == 1:
            axes = [axes]

        fig.suptitle(
            f"KL divergence between output of the full model and output of a circuit, for {self.results.experiment_name}, histogram"
        )

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
            ax.stairs(*torch.histogram(metrics, bins=100, range=range), label=circuit)

        for ax in axes:
            ax.set_xlabel("KL divergence")
            ax.set_ylabel("Frequency")
            ax.legend()

        plt.show()
        fig.savefig(figure_path := output_dir / "histogram.png")
        logger.info("Saved histogram to %s", figure_path)

    def get_experiment_data(self) -> AdvOptExperimentData:
        return get_standard_experiment_data(AdvOptTaskName[self.config["task"]["task_name"]])

    def analyse_all_metrics(self, experiment_data: AdvOptExperimentData | None = None) -> None:
        experiment_data = experiment_data or self.get_experiment_data()
        for circuit_name, metric in self.results.metrics.items():
            match circuit_name:
                case "canonical":
                    circuit = experiment_data.circuit_edges
                case _:
                    raise NotImplementedError()
            self.analyse_metrics(metric, circuit, experiment_data)

    def analyse_metrics(
        self, metrics: Float[torch.Tensor, " batch"], circuit: list[Edge], experiment_data: AdvOptExperimentData
    ) -> None:
        analyze_circuit_loss_metrics(
            experiment_data.masked_runner.masked_transformer.model.tokenizer,
            self.results.test_data,
            self.results.test_patch_data,
            metrics,
            circuit,
            experiment_data.masked_runner,
        )


def analyze_circuit_loss_metrics(
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
      - the loss obtained by those inputs
    """

    def decode(input: Integer[torch.Tensor, "batch pos"], map_depth: int = 0) -> str | list:
        if tokenizer is not None:
            return deep_map_with_depth(tokenizer.decode, input, map_depth)
        return input.tolist()

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

    # def calculate_entropy(logits: Float[torch.Tensor, "*batch vocab"]) -> Float[torch.Tensor, " *batch"]:
    #     p = torch.softmax(logits, dim=-1)
    #     return (-p * p.log()).sum(dim=-1)
    #
    # calculate_entropy(output_model[:, -1, :])
    # calculate_entropy(output_circuit[:, -1, :])

    for loss, input, patch_input, output_circuit, output_model in zip(
        topk_losses,
        decode(topk_most_adversarial_input, map_depth=1),
        decode(topk_most_adversarial_patch_input, map_depth=1),
        decode(top_k_most_likely_output_circuit.indices, map_depth=2),
        decode(top_k_most_likely_output_model.indices, map_depth=2),
    ):
        logger.info("------------------------------------")
        logger.info("Input: %s", input)
        logger.info("Patch input: %s", patch_input)
        logger.info("Loss: %s", loss.item())
        logger.info("Most likely output from circuit: %s", output_circuit)
        logger.info("Most likely output from model: %s", output_model)

    # fmt: off
    logger.info("Top %d loss: %s", top_k_input, topk_losses.tolist())
    logger.info("Top %d most adversarial input: %s", top_k_input, decode(topk_most_adversarial_input, map_depth=1))
    logger.info("Top %d most adversarial patch input: %s", top_k_input, decode(topk_most_adversarial_patch_input, map_depth=1))
    logger.info("Top %d most adversarial output from circuit: %s", top_k_out, decode(top_k_most_likely_output_circuit.indices, map_depth=2))
    logger.info("Top %d most adversarial output from model: %s", top_k_out, decode(top_k_most_likely_output_model.indices, map_depth=2))
    # fmt: on
