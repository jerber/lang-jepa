import torch
from torch import Tensor


def masked_mean(hidden: Tensor, attention_mask: Tensor) -> Tensor:
    """Mean-pool token embeddings along the sequence dim, excluding padding.

    Args:
        hidden: [batch, seq_len, dim] token-level features.
        attention_mask: [batch, seq_len] with 1 for real tokens, 0 for padding.

    Returns:
        [batch, dim] pooled embeddings.
    """
    mask = attention_mask.to(hidden.dtype).unsqueeze(-1)
    summed = (hidden * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1.0)
    return summed / denom
