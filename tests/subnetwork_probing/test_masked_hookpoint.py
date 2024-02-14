import torch
from transformer_lens.HookedTransformer import (
    HookedTransformer as LegacyHookedTransformer,
)
from transformer_lens.HookedTransformerConfig import (
    HookedTransformerConfig as LegacyHookedTransformerConfig,
)

from acdc.docstring.utils import get_all_docstring_things
from subnetwork_probing.train import MaskedTransformer


def do_random_resample_caching(model: LegacyHookedTransformer, train_data: torch.Tensor) -> torch.Tensor:
    for layer in model.blocks:
        layer.attn.hook_q.is_caching = True
        layer.attn.hook_k.is_caching = True
        layer.attn.hook_v.is_caching = True
        layer.hook_mlp_out.is_caching = True

    with torch.no_grad():
        outs = model(train_data)

    for layer in model.blocks:
        layer.attn.hook_q.is_caching = False
        layer.attn.hook_k.is_caching = False
        layer.attn.hook_v.is_caching = False
        layer.hook_mlp_out.is_caching = False

    return outs


def test_induction_mask_reimplementation_correct():
    all_task_things = get_all_docstring_things(
        num_examples=6,
        seq_len=41,
        device=torch.device("cpu"),
        metric_name="kl_div",
        correct_incorrect_wandb=False,
    )

    kwargs = dict(**all_task_things.tl_model.cfg.__dict__)
    for kwarg_string in [
        "use_split_qkv_input",
        "n_devices",
        "gated_mlp",
        "use_attn_in",
        "use_hook_mlp_in",
        "default_prepend_bos",
        "dtype",
        "add_special_tokens",
    ]:
        if kwarg_string in kwargs:
            del kwargs[kwarg_string]

    cfg = LegacyHookedTransformerConfig(**kwargs)
    # Create a model using the old version of SP, which uses MaskedHookPoints and a modified TransformerLens
    legacy_model = LegacyHookedTransformer(cfg, is_masked=True)
    legacy_model.load_state_dict(all_task_things.tl_model.state_dict(), strict=False)

    model = MaskedTransformer(all_task_things.tl_model)

    # Cache the un-patched data in each MaskedHookPoint
    _ = do_random_resample_caching(legacy_model, all_task_things.validation_patch_data)
    # Cache all the activations for the new model
    model.do_random_resample_caching(all_task_things.validation_patch_data)

    rng_state = torch.get_rng_state()
    with torch.no_grad():
        torch.set_rng_state(rng_state)
        out_legacy = legacy_model(all_task_things.validation_data)

        torch.set_rng_state(rng_state)
        with model.with_fwd_hooks_and_new_cache() as masked_model:
            out_hooks = masked_model(all_task_things.validation_data)

    assert torch.allclose(out_legacy, out_hooks)


def test_cache_writeable_forward_pass():
    all_task_things = get_all_docstring_things(
        num_examples=6,
        seq_len=41,
        device=torch.device("cpu"),
        metric_name="kl_div",
        correct_incorrect_wandb=False,
    )
    masked_model = MaskedTransformer(all_task_things.tl_model)

    # Test goes here

    # Run the model once on an unmodified cache
    rng_state = torch.random.get_rng_state()
    masked_model.do_random_resample_caching(all_task_things.validation_patch_data)
    context_args = dict(ablation="resample", ablation_data=all_task_things.validation_patch_data)
    with masked_model.with_fwd_hooks_and_new_cache(**context_args) as hooked_model:
        out1 = hooked_model(all_task_things.validation_data)

    # Now modify the cache and do it again
    torch.random.set_rng_state(rng_state)
    masked_model.do_random_resample_caching(all_task_things.validation_patch_data)
    for name in masked_model.ablation_cache:
        # We can't modify ActivationCache items, so we modify the underlying dict.
        masked_model.ablation_cache.cache_dict[name] = 1 - masked_model.ablation_cache[name]

    with masked_model.with_fwd_hooks_and_new_cache(**context_args) as hooked_model:
        out2 = hooked_model(all_task_things.validation_data)

    # We don't use allclose; the values should be exactly the same
    assert (out1 == out2).all()
