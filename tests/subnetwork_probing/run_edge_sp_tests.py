# %%

import torch

# from tests.subnetwork_probing.test_masked_hookpoint import test_cache_writeable_forward_pass
from acdc.tracr_task.utils import get_all_tracr_things, get_tracr_reverse_edges
from subnetwork_probing.train import MaskedTransformer

# %%

# all_task_things = get_all_docstring_things(
#     num_examples=6,
#     seq_len=41,
#     device=torch.device("cpu"),
#     metric_name="kl_div",
#     correct_incorrect_wandb=False,
# )
# %%

# test_cache_writeable_forward_pass()
# %%
all_tracr_things = get_all_tracr_things(task="reverse", metric_name="l2", num_examples=30, device=torch.device("cpu"))
masked_model = MaskedTransformer(all_tracr_things.tl_model, use_pos_embed=True)
# %%
masked_model.do_zero_caching()
# rng_state = torch.random.get_rng_state()
masked_model.do_random_resample_caching(all_tracr_things.validation_patch_data)
context_args = dict(ablation="resample", ablation_data=all_tracr_things.validation_patch_data)
with masked_model.with_fwd_hooks_and_new_cache(**context_args) as hooked_model:
    out1 = hooked_model(all_tracr_things.validation_data)

print(masked_model.forward_cache.cache_dict.keys())
# %%
# %%


def test_f_cache_implementation():
    """
    Verifies that the network running on forward cache with nothing resampled is
    identical to the unhooked network
    """
    global all_tracr_things
    # global masked_model

    masked_model = MaskedTransformer(all_tracr_things.tl_model, no_ablate=True)
    context_args = dict(ablation="resample", ablation_data=all_tracr_things.validation_patch_data)
    out1 = masked_model.model(all_tracr_things.validation_data)
    with masked_model.with_fwd_hooks_and_new_cache(**context_args) as hooked_model:
        out2 = hooked_model(all_tracr_things.validation_data)

    assert torch.allclose(out1, out2)
    print(f"Outputs of shape {out1.shape} are close:")


test_f_cache_implementation()
# %%


def test_reverse_gt_correct():
    """
    Tests that the ground truth circuit is a complete description of the model on the reverse task
    """
    all_task_things = get_all_tracr_things(
        task="reverse", metric_name="l2", num_examples=30, device=torch.device("cpu")
    )
    gt_edges = get_tracr_reverse_edges()
    masked_model = MaskedTransformer(all_task_things.tl_model, use_pos_embed=True)
    masked_model.freeze_weights()

    # Zero out the model logits...
    for logit_name in masked_model._mask_logits_dict:
        masked_model._mask_logits_dict[logit_name].data.fill_(-5)
    # Put the ground truth edges in the model logits...
    for c, ci, p, pi in gt_edges:
        # TODO deal with multiple heads
        if c not in masked_model.parent_node_names:
            print(f"SKIPPING edge {c}, {ci}, {p}, {pi}")
            continue
        print(f"checking edge {c}, {ci}, {p}, {pi}")
        p_index = masked_model.parent_node_names[c].index(p)
        # TODO deal with ci, pi
        masked_model._mask_logits_dict[c].data[p_index] = 5

    n_active_logits = sum((l >= 0).sum().item() for l in masked_model._mask_logits_dict.values())
    n_total_logits = sum(l.numel() for l in masked_model._mask_logits_dict.values())
    print(f"Model has {n_active_logits}/{n_total_logits} active logits and {len(gt_edges)} ground truth edges")

    # Run the model once without hooks
    rng_state = torch.random.get_rng_state()
    out1 = masked_model.model(all_task_things.validation_data)

    # Now run the masked model
    masked_model.do_random_resample_caching(all_task_things.validation_patch_data)
    context_args = dict(ablation="resample", ablation_data=all_task_things.validation_patch_data)
    with masked_model.with_fwd_hooks_and_new_cache(**context_args) as hooked_model:
        out2 = hooked_model(all_task_things.validation_data)

    reg_loss = masked_model.regularization_loss()
    print(f"Regularization loss of GT is {reg_loss}")
    assert torch.allclose(out1, out2), "Outputs of the masked model and the unmasked model are not close"


test_reverse_gt_correct()
# %%


def test_empty_circuit():
    """
    With the empty circuit (everything is patched):
    - After running, the ablation and forward caches should match, except for hook_embed.
    - TODO The ablation cache should be the same as run_with_cache on the patched data
    - The output should match the output on patched data.
    - Regularization loss should be low.
    - Loss should be high.
    """
    all_tracr_things = get_all_tracr_things(
        task="reverse", metric_name="l2", num_examples=30, device=torch.device("cpu")
    )
    masked_model = MaskedTransformer(all_tracr_things.tl_model, use_pos_embed=True)
    masked_model.freeze_weights()

    for logit_name in masked_model._mask_logits_dict:
        masked_model._mask_logits_dict[logit_name].data.fill_(-10)

    context_args = dict(ablation="resample", ablation_data=all_tracr_things.validation_patch_data)
    # context_args = dict(ablation="zero")
    out1 = masked_model.model(all_tracr_things.validation_patch_data)
    with masked_model.with_fwd_hooks_and_new_cache(**context_args) as hooked_model:
        out2 = hooked_model(all_tracr_things.validation_data)

    assert masked_model.ablation_cache.keys() == masked_model.forward_cache.keys()
    matching_cache_keys = set()
    differing_cache_keys = set()
    for key in masked_model.ablation_cache.keys():
        abla, forw = masked_model.ablation_cache[key], masked_model.forward_cache[key]
        if torch.allclose(abla, forw):
            matching_cache_keys.add(key)
        else:
            differing_cache_keys.add(key)
    print(f"Matching cache keys: {matching_cache_keys}")
    print(f"Differing cache keys: {differing_cache_keys}")
    assert "hook_embed" in differing_cache_keys, "hook_embed should be different"
    assert len(differing_cache_keys) == 1, "Only hook_embed should be different"

    assert torch.allclose(out1, out2), "Outputs of the masked model and the unmasked model on patch data are not close"

    # labels=all_tracr_things.validation_metric.keywords['model_out'].argmax(dim=-1)
    specific_metric_term = all_tracr_things.validation_metric(out2)
    reg_loss = masked_model.regularization_loss()
    print(f"Regularization loss of resample is {reg_loss}")
    assert reg_loss < 0.1, "Resample has too high regularization loss"
    lambda_reg = 100
    total_loss = specific_metric_term + lambda_reg * reg_loss
    print(f"Metric loss of resample is {specific_metric_term:.6f}")
    print(f"Total loss of resample is {total_loss:.6f}")
    assert specific_metric_term > 0.1, "Resample has too low metric loss"


test_empty_circuit()

# %%


def test_ablate_output_circuit():
    """
    With the circuit that contains all edges EXCEPT edges to last layer resid_post:
    - After running, the forward cache should match run_with_cache on validation data.
    - The output should match output on patched data.
    - Metric loss should be high.
    """
    all_tracr_things = get_all_tracr_things(
        task="reverse", metric_name="l2", num_examples=30, device=torch.device("cpu")
    )
    masked_model = MaskedTransformer(all_tracr_things.tl_model, use_pos_embed=True)
    masked_model.freeze_weights()

    output_logit_name = f"blocks.{masked_model.model.cfg.n_layers - 1}.hook_resid_post"
    for logit_name in masked_model._mask_logits_dict:
        masked_model._mask_logits_dict[logit_name].data.fill_(10)
    masked_model._mask_logits_dict[output_logit_name].data.fill_(-10)

    out0, cache = masked_model.model.run_with_cache(all_tracr_things.validation_data)
    out1 = masked_model.model(all_tracr_things.validation_patch_data)
    context_args = dict(ablation="resample", ablation_data=all_tracr_things.validation_patch_data)
    with masked_model.with_fwd_hooks_and_new_cache(**context_args) as hooked_model:
        out2 = hooked_model(all_tracr_things.validation_data)

    assert masked_model.forward_cache.keys() < cache.keys()
    for key in masked_model.forward_cache.keys():
        assert torch.allclose(
            masked_model.forward_cache[key], cache[key]
        ), f"forward_cache[{key}] is not close to cache[{key}]"

    diff = (out1 - out2).abs().sum()
    n_different_elements = (out1 != out2).sum()
    assert torch.allclose(
        out1, out2
    ), f"Outputs of masked model and unmasked model on patch data are not close: total diff={diff:.6f} on {n_different_elements} elements"

    specific_metric_term = all_tracr_things.validation_metric(out2)
    assert specific_metric_term > 0.1, "Resample has too low metric loss"


test_ablate_output_circuit()
# %%


def test_2_edge_circuit():
    """
    Test a particular 2-edge circuit found by SP; is it valid?
    """
    edge_names = [
        ("blocks.3.hook_v_input", "hook_embed"),
        # ("blocks.2.hook_mlp_in", "blocks.1.attn.hook_result"),
        ("blocks.3.hook_resid_post", "blocks.3.attn.hook_result"),
    ]
    all_task_things = get_all_tracr_things(
        task="reverse", metric_name="l2", num_examples=30, device=torch.device("cpu")
    )
    masked_model = MaskedTransformer(all_task_things.tl_model, use_pos_embed=True)
    masked_model.freeze_weights()

    for logit_name in masked_model._mask_logits_dict:
        masked_model._mask_logits_dict[logit_name].data.fill_(-10)

    for c, p in edge_names:
        c_index = masked_model.parent_node_names[c].index(p)
        masked_model._mask_logits_dict[c].data[c_index] = 10

    out1 = masked_model.model(all_tracr_things.validation_data)
    context_args = dict(ablation="resample", ablation_data=all_tracr_things.validation_patch_data)
    # context_args = dict(ablation='zero')
    with masked_model.with_fwd_hooks_and_new_cache(**context_args) as hooked_model:
        out2 = hooked_model(all_tracr_things.validation_data)

    diff = (out1 - out2).abs().sum()
    n_different_elements = (out1 != out2).sum()
    assert torch.allclose(
        out1, out2
    ), f"Outputs of the masked model and the unmasked model are not close: total diff={diff:.6f} on {n_different_elements} elements"


test_2_edge_circuit()
# %%
