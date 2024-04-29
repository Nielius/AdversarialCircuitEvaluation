from dataclasses import dataclass

import torch
from jaxtyping import Float, Shaped
from typing_extensions import Self

from acdc.nudb.adv_opt.analysis.output_parser import AdvOptHydraOutputDir, HydraConfig
from acdc.nudb.adv_opt.data_fetchers import AdvOptExperimentData, get_standard_experiment_data
from acdc.nudb.adv_opt.settings import ExperimentArtifacts
from acdc.nudb.adv_opt.utils import deep_map


@dataclass
class AdvOptBaseAnalyzer:
    artifacts: ExperimentArtifacts
    config: HydraConfig

    @classmethod
    def from_dir(cls, output_dir: AdvOptHydraOutputDir) -> Self:
        return cls(
            artifacts=output_dir.artifacts,
            config=output_dir.config,
        )

    def get_experiment_data(self) -> AdvOptExperimentData:
        return get_standard_experiment_data(task_name=self.config.task_name)

    def get_embedder(self) -> torch.nn.Module:
        experiment_data = get_standard_experiment_data(task_name=self.config.task_name)
        return experiment_data.masked_runner.masked_transformer.model.embed

    def get_tokenizer(self):
        experiment_data = get_standard_experiment_data(task_name=self.config.task_name)
        return experiment_data.masked_runner.masked_transformer.model.tokenizer


@dataclass
class AdvOptInputOnlyAnalyzer(AdvOptBaseAnalyzer):
    """Analyses the output for AdvOpt experiments that only optimize the input, while fixing the patch input."""

    def topk_coefficients(self, k: int = 10, final_coefficients: bool = True) -> torch.return_types.topk:
        """
        Return the base inputs with the top k highest coefficients among the final coefficients.

        Use the initial coefficients instead of the final coefficients if final_coefficients is False."""
        coeffs = self.artifacts.coefficients_final if final_coefficients else self.artifacts.coefficients_init
        assert coeffs is not None
        return torch.topk(coeffs, k)

    def topk_inputs_decoded(self, tokenizer, k: int = 10, final_coefficients: bool = True) -> list[str]:
        topk = self.topk_coefficients(k, final_coefficients)
        assert self.artifacts.base_input is not None
        return [tokenizer.decode(token) for token in self.artifacts.base_input[topk.indices, :]]

    def outputs_for_topk_inputs(
        self, experiment_data: AdvOptExperimentData, k: int = 10, circuit: bool = True, final_coefficients: bool = True
    ) -> Float[torch.Tensor, "batch pos vocab"]:
        """Get output for full model if `circuit` is set to False."""
        topk = self.topk_coefficients(k, final_coefficients)
        # Decode input:
        # [experiment_data.tokenizer.decode(token) for token in experiment_data.task_data.test_patch_data[0, ...]]
        assert self.artifacts.base_input is not None
        return experiment_data.masked_runner.run(
            input=self.artifacts.base_input[topk.indices, :],
            patch_input=experiment_data.task_data.test_patch_data[0, ...],
            edges_to_ablate=list(experiment_data.ablated_edges) if circuit else [],
        )

    def outputs_for_topk_inputs_decoded(
        self,
        experiment_data: AdvOptExperimentData,
        k_inputs: int = 10,
        k_outputs: int = 10,
        circuit: bool = True,
        final_coefficients: bool = True,
    ) -> Float[torch.Tensor, "batch pos vocab"]:
        """For each of the k_inputs worst inputs, get the k_outputs most likely output values, decoded."""
        outputs = self.outputs_for_topk_inputs(
            experiment_data, k_inputs, circuit, final_coefficients=final_coefficients
        )
        return deep_map(experiment_data.tokenizer.decode, torch.topk(outputs[:, -1, :], k=k_outputs).indices.tolist())

    def loss_for_topk_inputs(
        self,
        experiment_data: AdvOptExperimentData,
        k: int = 10,
    ) -> Float[torch.Tensor, " batch"]:
        outputs_circuit = self.outputs_for_topk_inputs(experiment_data, k, circuit=True)
        outputs_full_model = self.outputs_for_topk_inputs(experiment_data, k, circuit=False)
        return experiment_data.loss_fn(outputs_circuit, outputs_full_model)


def multidimensional_cartesian_split(
    t1: Shaped[torch.Tensor, " batch1 *rest"], t2: Shaped[torch.Tensor, " batch2 *rest"]
) -> tuple[Shaped[torch.Tensor, " batch1 batch2 *rest"], Shaped[torch.Tensor, " batch1 batch2 *rest"]]:
    xs, ys = torch.meshgrid(torch.arange(t1.shape[0]), torch.arange(t2.shape[0]), indexing="ij")
    return t1[xs, ...], t2[ys, ...]


def multidimensional_cartesian(
    t1: Shaped[torch.Tensor, " batch1 *d"], t2: Shaped[torch.Tensor, " batch2 *d"]
) -> Shaped[torch.Tensor, " batch1 batch2 *d 2"]:
    return torch.stack(multidimensional_cartesian_split(t1, t2), dim=-1)


@dataclass
class AdvOptAnalyzer(AdvOptBaseAnalyzer):
    """Analyses the output for AdvOpt experiments that optimize both input and patch data."""

    def topk_coefficients(
        self, k: int = 10, final_coefficients: bool = True
    ) -> tuple[torch.return_types.topk, torch.return_types.topk]:
        """
        Return the top k highest coefficients among the final coefficients.

        Use the initial coefficients instead of the final coefficients if final_coefficients is False."""
        coeffs = self.artifacts.coefficients_final if final_coefficients else self.artifacts.coefficients_init
        assert coeffs is not None

        coeffs_patch = (
            self.artifacts.coefficients_final_patch if final_coefficients else self.artifacts.coefficients_init_patch
        )
        assert coeffs_patch is not None

        return torch.topk(coeffs, k), torch.topk(coeffs_patch, k)

    def topk_inputs(self, k: int = 10, final_coefficients: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return the base inputs with the top k highest coefficients among the final coefficients.

        Use the initial coefficients instead of the final coefficients if final_coefficients is False."""
        topk, topk_patch = self.topk_coefficients(k, final_coefficients)
        assert self.artifacts.base_input is not None
        assert self.artifacts.base_patch_input is not None

        return (
            self.artifacts.base_input[topk.indices, :],
            self.artifacts.base_patch_input[topk_patch.indices, :],
        )

    def topk_inputs_decoded(
        self, tokenizer, k: int = 10, final_coefficients: bool = True
    ) -> tuple[list[str], list[str]]:
        topk, topk_patch = self.topk_inputs(k, final_coefficients)

        return (
            [tokenizer.decode(token) for token in topk],
            [tokenizer.decode(token) for token in topk_patch],
        )

    def outputs_for_topk_inputs(
        self, experiment_data: AdvOptExperimentData, k: int = 10, circuit: bool = True, final_coefficients: bool = True
    ) -> Float[torch.Tensor, "batch pos vocab"]:
        """Get output for full model if `circuit` is set to False."""
        topk_inputs, topk_inputs_patch = self.topk_inputs(k, final_coefficients)
        topk_inputs_combinations, topk_inputs_patch_combinations = multidimensional_cartesian_split(
            topk_inputs, topk_inputs_patch
        )

        return experiment_data.masked_runner.run(
            input=topk_inputs_combinations.flatten(start_dim=0, end_dim=1),
            patch_input=topk_inputs_patch_combinations.flatten(start_dim=0, end_dim=1),
            edges_to_ablate=list(experiment_data.ablated_edges) if circuit else [],
        )

    def outputs_for_topk_inputs_decoded(
        self,
        experiment_data: AdvOptExperimentData,
        k_inputs: int = 10,
        k_outputs: int = 10,
        circuit: bool = True,
        final_coefficients: bool = True,
    ) -> Float[torch.Tensor, "batch pos vocab"]:
        """For each of the k_inputs worst inputs, get the k_outputs most likely output values, decoded."""

        outputs = self.outputs_for_topk_inputs(
            experiment_data, k_inputs, circuit, final_coefficients=final_coefficients
        )
        return deep_map(experiment_data.tokenizer.decode, torch.topk(outputs[:, -1, :], k=k_outputs).indices.tolist())

    def loss_for_topk_inputs(
        self,
        experiment_data: AdvOptExperimentData,
        k: int = 10,
    ) -> Float[torch.Tensor, " batch"]:
        outputs_circuit = self.outputs_for_topk_inputs(experiment_data, k, circuit=True)
        outputs_full_model = self.outputs_for_topk_inputs(experiment_data, k, circuit=False)
        return experiment_data.loss_fn(outputs_circuit, outputs_full_model)
