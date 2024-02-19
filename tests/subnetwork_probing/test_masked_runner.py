import pytest
import torch

from acdc.docstring.utils import get_all_docstring_things
from acdc.greaterthan.utils import get_all_greaterthan_things
from acdc.nudb.adv_opt.data_fetchers import AdvOptTaskName
from acdc.nudb.adv_opt.masked_runner import MaskedRunner
from acdc.tracr_task.utils import get_all_tracr_things


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
