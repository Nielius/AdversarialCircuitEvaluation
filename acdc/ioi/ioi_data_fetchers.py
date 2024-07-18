import random
from dataclasses import dataclass
from functools import cache, partial

import torch
from jaxtyping import Int64
from torch.nn import functional as F
from transformer_lens import HookedTransformer

from acdc.acdc_utils import MatchNLLMetric, frac_correct_metric, kl_divergence, logit_diff_metric, negative_log_probs
from acdc.docstring.utils import AllDataThings
from acdc.ioi.ioi_dataset import IOIDataset
from acdc.ioi.ioi_dataset_constants import IOI_OBJECTS_DICT, NAMES
from acdc.ioi.ioi_dataset_v2 import (
    IOIPromptCollection,
    IOIPromptTemplate,
    generate_prompts_uniform_v2,
    get_ioi_tokenizer,
)
from acdc.ioi.utils import get_gpt2_small
from acdc.nudb.adv_opt.utils import model_forward_in_batches


@dataclass
class IOIExperimentInputData:
    """The input data that is split into validation and test data, and used to define the metrics."""

    num_examples: int

    default_data: Int64[torch.Tensor, "batch seq"]
    patch_data: Int64[torch.Tensor, "batch seq"]
    labels: Int64[torch.Tensor, " batch"]
    wrong_labels: Int64[torch.Tensor, " batch"]

    @property
    def validation_data(
        self,
    ) -> tuple[
        Int64[torch.Tensor, "batch seq"],
        Int64[torch.Tensor, "batch seq"],
        Int64[torch.Tensor, " batch"],
        Int64[torch.Tensor, " batch"],
    ]:
        return (
            self.default_data[: self.num_examples, :],
            self.patch_data[: self.num_examples, :],
            self.labels[: self.num_examples],
            self.wrong_labels[: self.num_examples],
        )

    @property
    def test_data(
        self,
    ) -> tuple[
        Int64[torch.Tensor, "batch seq"],
        Int64[torch.Tensor, "batch seq"],
        Int64[torch.Tensor, " batch"],
        Int64[torch.Tensor, " batch"],
    ]:
        return (
            self.default_data[self.num_examples :, :],
            self.patch_data[self.num_examples :, :],
            self.labels[self.num_examples :],
            self.wrong_labels[self.num_examples :],
        )

    @staticmethod
    def generate(
        rng: random.Random,
        num_examples: int,
        template: IOIPromptTemplate,
        patch_template: IOIPromptTemplate,
        names: list[str],
        template_values: dict[str, list[str]],
        device: str,
    ) -> "IOIExperimentInputData":
        base_dataset = IOIPromptCollection(
            prompts=generate_prompts_uniform_v2(
                template=template,
                template_values=template_values,
                names=names,
                num_prompts=2 * num_examples,
                rng=rng,
            ),
            num_examples=2 * num_examples,  # half is for validation, half is for test
        )
        patch_dataset = base_dataset.patch_random(
            rng,
            new_template=patch_template,
            new_template_values={},  # this keeps everything but the names the same
            names=names,
        )

        tokenizer = get_ioi_tokenizer()

        base_tokens = base_dataset.tokens(tokenizer, device=device).long()
        patch_tokens = patch_dataset.tokens(tokenizer, device=device).long()

        seq_len = base_tokens.shape[1]
        assert patch_tokens.shape[1] == seq_len, "Is this going to give problems?"

        default_data = base_tokens[: num_examples * 2, : seq_len - 1]
        patch_data = patch_tokens[: num_examples * 2, : seq_len - 1]
        labels = base_tokens[: num_examples * 2, seq_len - 1]
        wrong_labels = torch.as_tensor(base_dataset.s_tokenIDs(tokenizer), dtype=torch.long, device=device)

        assert torch.equal(
            labels, torch.as_tensor(base_dataset.io_tokenIDs(tokenizer), dtype=torch.long, device=device)
        )

        return IOIExperimentInputData(
            num_examples=num_examples,
            default_data=default_data,
            patch_data=patch_data,
            labels=labels,
            wrong_labels=wrong_labels,
        )


def get_all_ioi_things(num_examples, device, metric_name, kl_return_one_element=True) -> AllDataThings:
    input_data = get_input_for_all_ioi_things(device, num_examples)
    tl_model = get_gpt2_small(device=device)
    return get_all_ioi_things_for_input_data(
        input_data, tl_model, num_examples, device, metric_name, kl_return_one_element
    )


def get_all_ioi_things_for_input_data(
    input_data: IOIExperimentInputData,
    tl_model: HookedTransformer,
    num_examples,
    device,
    metric_name,
    kl_return_one_element=True,
) -> AllDataThings:
    validation_data, validation_patch_data, validation_labels, validation_wrong_labels = input_data.validation_data
    test_data, test_patch_data, test_labels, test_wrong_labels = input_data.test_data

    if metric_name == "none":
        # For many experiments, such as the brute force distribution analysis, we don't need any of these metrics.
        # The reason I disabled them, is that they take up a huge amount of memory, because the tensors have shape
        # (num_examples * 2, seq_len, vocab_size), and the vocab size is pretty big.
        def validation_metric():
            raise NotImplementedError("No metric specified")

        test_metrics = {}
    else:
        with torch.no_grad():
            base_model_logits = model_forward_in_batches(
                tl_model, input_data.default_data, batch_size=1024, slice_obj=(slice(None), -1, slice(None))
            )  # slice_obj=[:, -1, :]
            base_model_logprobs = F.log_softmax(base_model_logits, dim=-1)

        base_validation_logprobs = base_model_logprobs[:num_examples, :]
        base_test_logprobs = base_model_logprobs[num_examples:, :]

        if metric_name == "kl_div":
            validation_metric = partial(
                kl_divergence,
                base_model_logprobs=base_validation_logprobs,
                last_seq_element_only=True,
                base_model_probs_last_seq_element_only=False,
                return_one_element=kl_return_one_element,
            )
        elif metric_name == "logit_diff":
            validation_metric = partial(
                logit_diff_metric,
                correct_labels=validation_labels,
                wrong_labels=validation_wrong_labels,
            )
        elif metric_name == "frac_correct":
            validation_metric = partial(
                frac_correct_metric,
                correct_labels=validation_labels,
                wrong_labels=validation_wrong_labels,
            )
        elif metric_name == "nll":
            validation_metric = partial(
                negative_log_probs,
                labels=validation_labels,
                last_seq_element_only=True,
            )
        elif metric_name == "match_nll":
            validation_metric = MatchNLLMetric(
                labels=validation_labels,
                base_model_logprobs=base_validation_logprobs,
                last_seq_element_only=True,
            )
        else:
            raise ValueError(f"metric_name {metric_name} not recognized")

        test_metrics = {
            "kl_div": partial(
                kl_divergence,
                base_model_logprobs=base_test_logprobs,
                last_seq_element_only=True,
                base_model_probs_last_seq_element_only=False,
            ),
            "logit_diff": partial(
                logit_diff_metric,
                correct_labels=test_labels,
                wrong_labels=test_wrong_labels,
            ),
            "frac_correct": partial(
                frac_correct_metric,
                correct_labels=test_labels,
                wrong_labels=test_wrong_labels,
            ),
            "nll": partial(
                negative_log_probs,
                labels=test_labels,
                last_seq_element_only=True,
            ),
            "match_nll": MatchNLLMetric(
                labels=test_labels,
                base_model_logprobs=base_test_logprobs,
                last_seq_element_only=True,
            ),
        }

    return AllDataThings(
        tl_model=tl_model,
        validation_metric=validation_metric,
        validation_data=validation_data,
        validation_labels=validation_labels,
        validation_mask=None,
        validation_patch_data=validation_patch_data,
        test_metrics=test_metrics,
        test_data=test_data,
        test_labels=test_labels,
        test_mask=None,
        test_patch_data=test_patch_data,
    )


def get_input_for_all_ioi_things(device, num_examples: int) -> IOIExperimentInputData:
    ioi_dataset = IOIDataset(prompt_type="ABBA", seed=0, N=num_examples * 2, num_templates=1)
    abc_dataset = (
        ioi_dataset.gen_flipped_prompts(("IO", "RAND"), seed=1)
        .gen_flipped_prompts(("S", "RAND"), seed=2)
        .gen_flipped_prompts(("S1", "RAND"), seed=3)
    )

    seq_len = ioi_dataset.toks.shape[1]
    assert seq_len == 16, f"Well, I thought ABBA #1 was 16 not {seq_len} tokens long..."

    default_data = ioi_dataset.toks.long()[: num_examples * 2, : seq_len - 1].to(device)
    patch_data = abc_dataset.toks.long()[: num_examples * 2, : seq_len - 1].to(device)
    labels = ioi_dataset.toks.long()[: num_examples * 2, seq_len - 1]
    wrong_labels = torch.as_tensor(ioi_dataset.s_tokenIDs[: num_examples * 2], dtype=torch.long, device=device)

    assert torch.equal(labels, torch.as_tensor(ioi_dataset.io_tokenIDs, dtype=torch.long))
    labels = labels.to(device)

    return IOIExperimentInputData(
        num_examples=num_examples,
        default_data=default_data,
        patch_data=patch_data,
        labels=labels,
        wrong_labels=wrong_labels,
    )


class IOIExperimentDataGenerator:
    """New version of the 'get_all_ioi_things'."""

    _device: str
    metric_name: str
    kl_return_one_element: bool
    num_examples: int

    def __init__(self, num_examples: int, device: str, metric_name: str, kl_return_one_element: bool = True):
        self._device = device
        self.metric_name = metric_name
        self.kl_return_one_element = kl_return_one_element
        self.num_examples = num_examples

    @cache
    def get_model(self) -> HookedTransformer:
        return get_gpt2_small(device=self._device)

    def get_input_data(self, rng: random.Random, template: IOIPromptTemplate) -> IOIExperimentInputData:
        # Assumptions:
        # 1. Can generate appropriate patch data with ABCA
        # 2. The subject and indirect object positions are the same in the patch template as in the original
        assert len(template.names_order) == 4, "At the moment, we only support templates with 4 names"

        return IOIExperimentInputData.generate(
            rng=rng,
            num_examples=self.num_examples,
            template=template,
            patch_template=template.with_different_order(list("ABCA")),
            names=NAMES,
            template_values=IOI_OBJECTS_DICT,
            device=self._device,
        )

    def get_all_data_things(self, rng: random.Random, template: IOIPromptTemplate) -> AllDataThings:
        return get_all_ioi_things_for_input_data(
            self.get_input_data(rng, template),
            self.get_model(),
            self.num_examples,
            self._device,
            self.metric_name,
            self.kl_return_one_element,
        )
