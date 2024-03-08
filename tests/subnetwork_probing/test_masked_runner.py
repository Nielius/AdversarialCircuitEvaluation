import pytest
import torch
from jaxtyping import Float

from acdc.docstring.utils import AllDataThings, get_all_docstring_things
from acdc.greaterthan.utils import get_all_greaterthan_things
from acdc.nudb.adv_opt.data_fetchers import EXPERIMENT_DATA_PROVIDERS, AdvOptExperimentData, AdvOptTaskName
from acdc.tracr_task.utils import get_all_tracr_things
from subnetwork_probing.masked_transformer import create_mask_parameters_and_forward_cache_hook_points


@pytest.fixture(scope="class")
def experiment_data_fixture(request) -> AdvOptExperimentData:
    task_name = request.param
    return EXPERIMENT_DATA_PROVIDERS[task_name].get_experiment_data(
        num_examples=6 if task_name != AdvOptTaskName.TRACR_REVERSE else 30,
        metric_name="kl_div" if task_name != AdvOptTaskName.TRACR_REVERSE else "l2",
        device="cpu",
    )


def get_all_task_things(task: AdvOptTaskName) -> AllDataThings:
    match task:
        case AdvOptTaskName.DOCSTRING:
            return get_all_docstring_things(
                num_examples=6,
                seq_len=41,
                device=torch.device("cpu"),
                metric_name="kl_div",
                correct_incorrect_wandb=False,
            )
        case AdvOptTaskName.TRACR_REVERSE:
            return get_all_tracr_things(
                task="reverse",
                num_examples=30,
                device=torch.device("cpu"),
                metric_name="l2",
            )
        case AdvOptTaskName.GREATERTHAN:
            return get_all_greaterthan_things(
                num_examples=6,
                device=torch.device("cpu"),
                metric_name="kl_div",
            )
        case _:
            raise NotImplementedError()


@pytest.mark.parametrize(
    "experiment_data_fixture",
    [AdvOptTaskName.DOCSTRING, AdvOptTaskName.TRACR_REVERSE, AdvOptTaskName.GREATERTHAN],
    indirect=True,
    scope="class",
)
class TestMaskedRunner:
    def test_running_without_ablating_edges_is_same_as_running_underlying_model(
        self, experiment_data_fixture: AdvOptExperimentData
    ):
        # We need at least GREATERTHAN, because that has got mlp and attention,
        # and the tracr seems to be too small (?) to be able to pick up some mistakes.
        masked_runner = experiment_data_fixture.masked_runner
        all_task_things_fixture = experiment_data_fixture.task_data

        rng_state = torch.random.get_rng_state()
        output_tl_model = all_task_things_fixture.tl_model(all_task_things_fixture.validation_data)
        torch.random.set_rng_state(rng_state)
        output_full_circuit = masked_runner.run(
            all_task_things_fixture.validation_data, all_task_things_fixture.validation_patch_data, edges_to_ablate=[]
        )

        assert torch.allclose(
            output_tl_model, output_full_circuit, atol=1e-4
        )  # it looks like there are small discrepancies in each layer, that add up; not sure why we are getting the small discrepancies though

    def test_running_with_everything_ablated_is_same_as_running_underlying_model_on_patch(
        self, experiment_data_fixture: AdvOptExperimentData
    ):
        # We need at least GREATERTHAN, because that has got mlp and attention,
        # and the tracr seems to be too small (?) to be able to pick up some mistakes.
        masked_runner = experiment_data_fixture.masked_runner
        all_task_things_fixture = experiment_data_fixture.task_data

        rng_state = torch.random.get_rng_state()
        output_tl_model = all_task_things_fixture.tl_model(all_task_things_fixture.validation_patch_data)
        torch.random.set_rng_state(rng_state)
        output_full_circuit = masked_runner.run(
            all_task_things_fixture.validation_data,
            all_task_things_fixture.validation_patch_data,
            edges_to_ablate=list(masked_runner.all_ablatable_edges),
        )

        assert torch.allclose(
            output_tl_model, output_full_circuit, atol=1e-4
        )  # it looks like there are small discrepancies in each layer, that add up; not sure why we are getting the small discrepancies though

    def test_run_convex_combination_without_ablation(self, experiment_data_fixture: AdvOptExperimentData):
        """Test: run the model with a convex combination that just selects the 3rd input point.
        That should be the same as running the model on the 3rd input point."""
        masked_runner = experiment_data_fixture.masked_runner
        all_task_things_fixture = experiment_data_fixture.task_data
        data_point_index = 2  # with a linear combination we're just going to select the third point

        rng_state = torch.random.get_rng_state()
        output_tl_model: Float[torch.Tensor, "batch pos vocab"] = all_task_things_fixture.tl_model(
            all_task_things_fixture.validation_data[data_point_index, ...]
        )

        coefficients = torch.full(
            (all_task_things_fixture.validation_patch_data.shape[0],), float("-inf"), dtype=torch.float32
        )
        coefficients[data_point_index] = 0.0

        torch.random.set_rng_state(rng_state)
        output_masked_runner = masked_runner.run_with_linear_combination(
            input_embedded=masked_runner.masked_transformer.model.embed(all_task_things_fixture.validation_data),
            dummy_input=all_task_things_fixture.validation_data[0, ...],
            coefficients=coefficients,
            patch_input=all_task_things_fixture.validation_patch_data[0, ...],
            edges_to_ablate=[],
        )

        assert torch.allclose(output_tl_model, output_masked_runner, atol=1e-4)

    def test_run_convex_patch_combination_without_ablation(self, experiment_data_fixture: AdvOptExperimentData):
        """Test: run the model with a convex combination that just selects the 3rd input point.
        That should be the same as running the model on the 3rd input point."""
        masked_runner = experiment_data_fixture.masked_runner
        all_task_things_fixture = experiment_data_fixture.task_data
        data_point_index = 2  # with a linear combination we're just going to select the third point

        rng_state = torch.random.get_rng_state()
        output_tl_model: Float[torch.Tensor, "batch pos vocab"] = all_task_things_fixture.tl_model(
            all_task_things_fixture.validation_data[data_point_index, ...]
        )

        coefficients = torch.full(
            (all_task_things_fixture.validation_patch_data.shape[0],), float("-inf"), dtype=torch.float32
        )
        coefficients[data_point_index] = 0.0

        coefficients_patch = torch.rand((all_task_things_fixture.validation_patch_data.shape[0],), dtype=torch.float32)

        torch.random.set_rng_state(rng_state)
        output_masked_runner = masked_runner.run_with_linear_combination_of_input_and_patch(
            input_embedded=masked_runner.masked_transformer.model.embed(all_task_things_fixture.validation_data),
            patch_input_embedded=masked_runner.masked_transformer.model.embed(
                all_task_things_fixture.validation_patch_data
            ),
            dummy_input=all_task_things_fixture.validation_data[0, ...],
            coefficients=coefficients,
            coefficients_patch=coefficients_patch,
            edges_to_ablate=[],
        )

        assert torch.allclose(output_tl_model, output_masked_runner, atol=1e-4)

    def test_run_convex_combination_with_ablation(self, experiment_data_fixture: AdvOptExperimentData):
        """Test: run the model with a convex combination that just selects the 3rd input point.
        That should be the same as running the model on the 3rd input point."""
        masked_runner = experiment_data_fixture.masked_runner
        all_task_things = experiment_data_fixture.task_data
        data_point_index = 2  # with a linear combination we're just going to select the third point
        ablated_edges = list(experiment_data_fixture.ablated_edges)

        rng_state = torch.random.get_rng_state()
        output_masked_runner_normal = masked_runner.run(
            input=all_task_things.validation_data[0, ...].unsqueeze(0),
            patch_input=all_task_things.validation_patch_data[0, ...].unsqueeze(0),
            edges_to_ablate=ablated_edges,
        )

        coefficients = torch.full((all_task_things.validation_patch_data.shape[0],), float("-inf"), dtype=torch.float32)
        coefficients[data_point_index] = 0.0

        torch.random.set_rng_state(rng_state)
        output_masked_runner_convex_combination = masked_runner.run_with_linear_combination(
            input_embedded=masked_runner.masked_transformer.model.embed(all_task_things.validation_data),
            dummy_input=all_task_things.validation_data[0, ...],
            coefficients=coefficients,
            patch_input=all_task_things.validation_patch_data[0, ...],
            edges_to_ablate=ablated_edges,
        )

        assert torch.allclose(output_masked_runner_normal, output_masked_runner_convex_combination, atol=1e-4)

    def test_run_convex_patch_combination_stores_the_right_ablation_cache(
        self, experiment_data_fixture: AdvOptExperimentData
    ):
        """Test: run the model with a convex combination that just selects the 3rd input point.
        That should be the same as running the model on the 3rd input point."""
        masked_runner = experiment_data_fixture.masked_runner
        all_task_things_fixture = experiment_data_fixture.task_data
        data_point_index = 2  # with a linear combination we're just going to select the third point

        rng_state = torch.random.get_rng_state()
        coefficients = torch.rand((all_task_things_fixture.validation_patch_data.shape[0],), dtype=torch.float32)
        coefficients_patch = torch.full(
            (all_task_things_fixture.validation_patch_data.shape[0],), float("-inf"), dtype=torch.float32
        )
        coefficients_patch[data_point_index] = 0.0
        torch.random.set_rng_state(rng_state)

        # calculate cache using convex combination
        masked_runner.run_with_linear_combination_of_input_and_patch(
            input_embedded=masked_runner.masked_transformer.model.embed(all_task_things_fixture.validation_data),
            patch_input_embedded=masked_runner.masked_transformer.model.embed(
                all_task_things_fixture.validation_patch_data
            ),
            dummy_input=all_task_things_fixture.validation_data[0, ...],
            coefficients=coefficients,
            coefficients_patch=coefficients_patch,
            edges_to_ablate=[],
        )
        convex_combination_cache = masked_runner.masked_transformer.ablation_cache

        # calculate cache more directly
        masked_runner.masked_transformer.calculate_and_store_ablation_cache(
            all_task_things_fixture.validation_patch_data[data_point_index, ...]
        )
        direct_cache = masked_runner.masked_transformer.ablation_cache

        # assert that the two caches are the same
        assert len(convex_combination_cache) == len(direct_cache)
        for k, v in convex_combination_cache.items():
            assert torch.allclose(v, direct_cache[k], atol=1e-4)


def test_create_mask_parameters_and_forward_cache_hook_points():
    outputs = create_mask_parameters_and_forward_cache_hook_points(
        use_pos_embed=True,
        num_heads=4,
        num_layers=3,
        device="cpu",
        mask_init_constant=0.123,
        attn_only=False,
    )

    ordered_forward_cache_hook_points = outputs[0]
    hook_point_to_parents = outputs[1]
    # mask_parameter_list = outputs[2]
    # mask_parameter_dict = outputs[3]

    for parent_list in hook_point_to_parents.values():
        for parent in parent_list:
            assert parent in ordered_forward_cache_hook_points, f"{parent} not in forward cache names"

    assert "hook_embed" in ordered_forward_cache_hook_points
    assert "hook_pos_embed" in ordered_forward_cache_hook_points

    # could test the shapes here, but I'm going to leave it for now
