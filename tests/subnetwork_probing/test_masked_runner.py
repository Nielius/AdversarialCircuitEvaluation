import pytest
import torch

from acdc.docstring.utils import get_all_docstring_things
from acdc.greaterthan.utils import get_all_greaterthan_things
from acdc.nudb.adv_opt.data_fetchers import AdvOptTaskName
from acdc.nudb.adv_opt.masked_runner import MaskedRunner
from acdc.tracr_task.utils import get_all_tracr_things
from subnetwork_probing.masked_transformer import create_mask_parameters_and_forward_cache_hook_points


@pytest.mark.parametrize("task", [AdvOptTaskName.DOCSTRING, AdvOptTaskName.TRACR_REVERSE, AdvOptTaskName.GREATERTHAN])
def test_running_without_ablating_edges_is_same_as_running_underlying_model(task: AdvOptTaskName):
    # We need at least GREATERTHAN, because that has got mlp and attention,
    # and the tracr seems to be too small (?) to be able to pick up some mistakes.
    if task == AdvOptTaskName.DOCSTRING:
        all_task_things = get_all_docstring_things(
            num_examples=6,
            seq_len=41,
            device=torch.device("cpu"),
            metric_name="kl_div",
            correct_incorrect_wandb=False,
        )
    elif task == AdvOptTaskName.TRACR_REVERSE:
        all_task_things = get_all_tracr_things(
            task="reverse",
            num_examples=30,
            device=torch.device("cpu"),
            metric_name="l2",
        )
    elif task == AdvOptTaskName.GREATERTHAN:
        all_task_things = get_all_greaterthan_things(
            num_examples=6,
            device=torch.device("cpu"),
            metric_name="kl_div",
        )

    masked_runner = MaskedRunner(all_task_things.tl_model)

    rng_state = torch.random.get_rng_state()
    output_tl_model = all_task_things.tl_model(all_task_things.validation_data)
    torch.random.set_rng_state(rng_state)
    output_full_circuit = masked_runner.run(
        all_task_things.validation_data, all_task_things.validation_patch_data, edges_to_ablate=[]
    )

    assert torch.allclose(
        output_tl_model, output_full_circuit, atol=1e-4
    )  # it looks like there are small discrepancies in each layer, that add up; not sure why we are getting the small discrepancies though


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
