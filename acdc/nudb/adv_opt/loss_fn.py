import torch
from jaxtyping import Float
from torch.nn import functional as F


def kl_div_on_output_logits(
    base_output_logits: Float[torch.Tensor, "batch pos vocab"],
    other_output_logits: Float[torch.Tensor, "batch pos vocab"],
    last_sequence_position_only: bool,
) -> Float[torch.Tensor, " batch"]:
    assert base_output_logits.shape == other_output_logits.shape
    assert len(base_output_logits.shape) == 3  # batch, pos, vocab

    if last_sequence_position_only:
        # depending on the task, we only want to take the last sequence position or not.
        # E.g. for the reverse task, every sequence position matters.
        # But for e.g. the docstring task, we only want to get the metrics
        # from the final sequence position.
        metrics = F.kl_div(
            F.log_softmax(other_output_logits[:, -1, :], dim=-1),  # prediction
            F.log_softmax(base_output_logits[:, -1, :], dim=-1),  # true
            reduction="none",
            log_target=True,
        ).sum(dim=-1)
    else:
        metrics = (
            F.kl_div(
                F.log_softmax(other_output_logits, dim=-1),
                F.log_softmax(base_output_logits, dim=-1),
                reduction="none",
                log_target=True,
            )
            .sum(dim=-1)  # sum over vocab
            .mean(dim=-1)  # mean of sequence positions
        )
    return metrics
