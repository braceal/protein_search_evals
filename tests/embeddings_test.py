"""Tests for HDF5-backed token embedding storage."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from genslm_embeddings.embed.embeddings import HDF5TokenEmbeddings
from genslm_embeddings.embed.embeddings import TokenEmbedInfo


def test_token_embed_info_reports_shape_metadata() -> None:
    info = TokenEmbedInfo(
        sequences=['AAA', 'CC'],
        embeddings=[np.zeros((3, 4)), np.ones((2, 4))],
    )

    assert len(info) == 2
    assert info.embedding_dim == 4
    assert info.seq_lengths == [3, 2]


def test_empty_token_embed_info_has_no_embedding_dimension() -> None:
    with pytest.raises(ValueError, match='No embeddings'):
        TokenEmbedInfo().embedding_dim


def test_hdf5_embeddings_round_trip_ragged_arrays(tmp_path: Path) -> None:
    embeddings = [
        np.arange(6, dtype=np.float32).reshape(3, 2),
        np.array([[10.0, 11.0]], dtype=np.float32),
    ]
    store = HDF5TokenEmbeddings(
        tmp_path / 'embeddings.h5',
        buffer_size=2,
        max_sequence_length=4,
    )

    store.append(['AAA', 'C'], embeddings)

    assert len(store) == 2
    np.testing.assert_array_equal(store[0], embeddings[0])
    selected = store[[1, 0]]
    np.testing.assert_array_equal(selected[0], embeddings[1])
    np.testing.assert_array_equal(selected[1], embeddings[0])
    by_sequence = store.get_embeddings(['C', 'AAA'])
    np.testing.assert_array_equal(by_sequence[0], embeddings[1])
    np.testing.assert_array_equal(by_sequence[1], embeddings[0])
    store.close()


def test_hdf5_embeddings_flushes_partial_buffer(tmp_path: Path) -> None:
    store = HDF5TokenEmbeddings(
        tmp_path / 'partial.h5',
        buffer_size=10,
        max_sequence_length=2,
    )
    embedding = np.array([[1.0, 2.0]], dtype=np.float32)
    store.append(['A'], [embedding])

    assert not store.file.exists()

    store.flush()

    assert len(store) == 1
    np.testing.assert_array_equal(store[:][0], embedding)
    with pytest.raises(TypeError, match='Indexing with type'):
        store['A']  # type: ignore[index]
    store.close()
