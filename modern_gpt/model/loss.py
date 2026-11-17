"""Reusable loss calculation and token-weighted accumulation utilities for ModernGPT-S SFT.

Provides mathematically verified, token-weighted cross-entropy loss functions
that ensure exact equivalence between sequential microbatch gradient accumulation
and single-pass global effective batch execution.
"""

from typing import Optional, Tuple
import torch
import torch.nn.functional as F


def compute_token_weighted_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
    normalizer: Optional[float] = None,
) -> Tuple[torch.Tensor, int, float]:
    """Compute causal LM cross-entropy with exact token-weighted normalization.

    Args:
        logits: Unnormalized model predictions of shape (B, T, V).
        labels: Target token IDs of shape (B, T).
        ignore_index: Target token ID to exclude from loss (default: -100).
        normalizer: Denominator for loss scaling. If provided and > 0, the summed
            cross-entropy is divided by this value (e.g. N_total of the accumulation window).
            If None, divides by the number of valid supervised tokens in this batch.
            If normalizer is provided but <= 0 (e.g. window with 0 supervised tokens),
            returns a safe zero-loss tensor.

    Returns:
        loss: Scalar tensor representing the normalized cross-entropy.
        valid_tokens: Integer count of valid supervised tokens in this batch.
        loss_sum: Python float sum of cross-entropy over all valid tokens in this batch.
    """
    # Standard causal next-token alignment: position t predicts token at t+1
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    vocab_size = shift_logits.size(-1)
    flat_logits = shift_logits.view(-1, vocab_size)
    flat_labels = shift_labels.view(-1)

    valid_mask = flat_labels != ignore_index
    valid_tokens = int(valid_mask.sum().item())

    if valid_tokens == 0:
        # Zero supervised tokens in this microbatch.
        # Return 0.0 with gradient graph connected to flat_logits so backward does not fail.
        zero_loss = flat_logits.sum() * 0.0
        return zero_loss, 0, 0.0

    # Compute unreduced sum of negative log likelihoods over valid supervised tokens
    sum_loss = F.cross_entropy(
        flat_logits,
        flat_labels,
        ignore_index=ignore_index,
        reduction="sum",
    )
    loss_sum_val = float(sum_loss.item())

    if normalizer is not None:
        if normalizer <= 0:
            loss = flat_logits.sum() * 0.0
        else:
            loss = sum_loss / float(normalizer)
    else:
        loss = sum_loss / float(max(1, valid_tokens))

    return loss, valid_tokens, loss_sum_val

