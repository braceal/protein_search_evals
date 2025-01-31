"""Pooling functions for hidden states."""

from __future__ import annotations

import torch


# TODO: We might want to configure whether to include the start and end tokens
def average_pool(
    embeddings: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Average pool the hidden states using the attention mask.

    Parameters
    ----------
    embeddings : torch.Tensor
        The hidden states to pool (B, SeqLen, HiddenDim).
    attention_mask : torch.Tensor
        The attention mask for the hidden states (B, SeqLen).

    Returns
    -------
    torch.Tensor
        The pooled embeddings (B, HiddenDim).
    """
    # Clone the attention mask to avoid modifying the original tensor
    attn_mask = attention_mask.clone()

    # Get the sequence lengths
    seq_lengths = attn_mask.sum(axis=1)

    # Set the attention mask to 0 for start and end tokens
    attn_mask[:, 0] = 0
    attn_mask[:, seq_lengths - 1] = 0

    # Create a mask for the pooling operation (B, SeqLen, HiddenDim)
    pool_mask = attn_mask.unsqueeze(-1).expand(embeddings.shape)

    # Sum the embeddings over the sequence length (use the mask to avoid
    # pad, start, and stop tokens)
    sum_embeds = torch.sum(embeddings * pool_mask, 1)

    # Avoid division by zero for zero length sequences by clamping
    sum_mask = torch.clamp(pool_mask.sum(1), min=1e-9)

    # Compute mean pooled embeddings for each sequence
    return sum_embeds / sum_mask
