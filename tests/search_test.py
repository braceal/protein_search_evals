"""Tests for layer-aware embedding search."""

from __future__ import annotations

import numpy as np
import pytest

from genslm_embeddings.embed.writers import select_embedding_layer


def test_select_embedding_layer_uses_transformer_block_metadata() -> None:
    """Resolve public transformer block numbers to the stored layer axis."""
    embeddings = np.arange(24).reshape(2, 3, 4)
    metadata = {'layer_indices': [5, 2, 8]}

    selected = select_embedding_layer(embeddings, 2, metadata)

    np.testing.assert_array_equal(selected, embeddings[:, 1, :])


def test_select_embedding_layer_requires_layer_for_multilayer_data() -> None:
    """Never silently select a layer from a three-dimensional dataset."""
    with pytest.raises(ValueError, match='embedding_layer is required'):
        select_embedding_layer(
            np.ones((2, 3, 4)),
            None,
            {'layer_indices': [0, 1, 2]},
        )


def test_select_embedding_layer_rejects_unavailable_block() -> None:
    """Reject transformer block numbers absent from stored metadata."""
    with pytest.raises(ValueError, match=r'stored layers are \[0, 2\]'):
        select_embedding_layer(
            np.ones((2, 2, 4)),
            1,
            {'layer_indices': [0, 2]},
        )


def test_select_embedding_layer_preserves_final_only_data() -> None:
    """Return existing two-dimensional datasets without a selector."""
    embeddings = np.ones((2, 4))

    assert select_embedding_layer(embeddings, None, None) is embeddings
    with pytest.raises(ValueError, match='final-layer-only'):
        select_embedding_layer(embeddings, 0, None)


def test_select_embedding_layer_supports_nested_dataset_indices() -> None:
    """Select the layer axis after multi-dimensional dataset indexing."""
    embeddings = np.arange(48).reshape(2, 2, 3, 4)

    selected = select_embedding_layer(
        embeddings,
        5,
        {'layer_indices': [1, 5, 8]},
    )

    np.testing.assert_array_equal(selected, embeddings[:, :, 1, :])
