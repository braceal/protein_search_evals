"""Tests for embedding pooling functions."""

from __future__ import annotations

import torch

from genslm_embeddings.embed.poolers import average_pool


def test_average_pool_excludes_special_tokens_and_padding() -> None:
    embeddings = torch.tensor(
        [
            [[100.0], [2.0], [4.0], [200.0], [999.0]],
            [[100.0], [6.0], [200.0], [999.0], [999.0]],
        ],
    )
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1, 0],
            [1, 1, 1, 0, 0],
        ],
    )
    original_mask = attention_mask.clone()

    pooled = average_pool(embeddings, attention_mask)

    torch.testing.assert_close(pooled, torch.tensor([[3.0], [6.0]]))
    assert torch.equal(attention_mask, original_mask)


def test_average_pool_can_include_special_tokens() -> None:
    embeddings = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [8.0, 9.0]]])
    attention_mask = torch.tensor([[1, 1, 0]])

    pooled = average_pool(
        embeddings,
        attention_mask,
        sos_token=False,
        eos_token=False,
    )

    torch.testing.assert_close(pooled, torch.tensor([[2.0, 3.0]]))


def test_average_pool_returns_zero_when_only_special_tokens_remain() -> None:
    embeddings = torch.tensor([[[10.0], [20.0]]])
    attention_mask = torch.tensor([[1, 1]])

    pooled = average_pool(embeddings, attention_mask)

    torch.testing.assert_close(pooled, torch.zeros((1, 1)))
