"""Tests for embedding result writers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from datasets import Dataset

from genslm_embeddings.embed.writers import HuggingFaceWriter
from genslm_embeddings.embed.writers import load_embedding_metadata


def test_hugging_face_writer_writes_and_merges_datasets(
    tmp_path: Path,
) -> None:
    writer = HuggingFaceWriter()
    first_dir = tmp_path / 'first'
    second_dir = tmp_path / 'second'
    merged_dir = tmp_path / 'merged'

    writer.write(first_dir, {'sequence': ['AAA'], 'score': [1.0]})
    writer.write(
        second_dir,
        {'sequence': ['CCC', 'GGG'], 'score': [2.0, 3.0]},
    )
    writer.merge([first_dir, second_dir], merged_dir)

    merged = Dataset.load_from_disk(merged_dir)
    assert merged.to_dict() == {
        'sequence': ['AAA', 'CCC', 'GGG'],
        'score': [1.0, 2.0, 3.0],
    }


def test_hugging_face_writer_preserves_multilayer_metadata(
    tmp_path: Path,
) -> None:
    """Store sequence metadata once and preserve layer metadata on merge."""
    writer = HuggingFaceWriter()
    first_dir = tmp_path / 'first'
    second_dir = tmp_path / 'second'
    merged_dir = tmp_path / 'merged'
    metadata = {
        'hidden_dimension': 2,
        'layer_indices': [0, 2],
        'normalized': True,
    }

    writer.write(
        first_dir,
        {
            'embeddings': np.ones((1, 2, 2)),
            'sequences': ['AAA'],
            'tags': ['first'],
        },
        metadata=metadata,
    )
    writer.write(
        second_dir,
        {
            'embeddings': np.zeros((1, 2, 2)),
            'sequences': ['CCC'],
            'tags': ['second'],
        },
        metadata=metadata,
    )

    writer.merge([first_dir, second_dir], merged_dir)

    merged = Dataset.load_from_disk(merged_dir)
    assert merged.column_names == ['embeddings', 'sequences', 'tags']
    assert merged['sequences'] == ['AAA', 'CCC']
    assert merged['tags'] == ['first', 'second']
    assert np.asarray(merged['embeddings']).shape == (2, 2, 2)
    assert load_embedding_metadata(merged_dir) == metadata


def test_hugging_face_writer_rejects_mismatched_metadata(
    tmp_path: Path,
) -> None:
    """Do not merge shards produced with different layer selections."""
    writer = HuggingFaceWriter()
    first_dir = tmp_path / 'first'
    second_dir = tmp_path / 'second'

    writer.write(
        first_dir,
        {'embeddings': np.ones((1, 1, 2))},
        metadata={'layer_indices': [0]},
    )
    writer.write(
        second_dir,
        {'embeddings': np.ones((1, 1, 2))},
        metadata={'layer_indices': [1]},
    )

    with pytest.raises(ValueError, match='different metadata'):
        writer.merge([first_dir, second_dir], tmp_path / 'merged')


def test_hugging_face_writer_rejects_mismatched_embedding_shapes(
    tmp_path: Path,
) -> None:
    """Do not merge shards with different hidden dimensions."""
    writer = HuggingFaceWriter()
    first_dir = tmp_path / 'first'
    second_dir = tmp_path / 'second'
    metadata = {'layer_indices': [0, 1]}

    writer.write(
        first_dir,
        {'embeddings': np.ones((1, 2, 2))},
        metadata=metadata,
    )
    writer.write(
        second_dir,
        {'embeddings': np.ones((1, 2, 3))},
        metadata=metadata,
    )

    with pytest.raises(ValueError, match='different shapes'):
        writer.merge([first_dir, second_dir], tmp_path / 'merged')
