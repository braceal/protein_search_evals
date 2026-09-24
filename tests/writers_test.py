"""Tests for embedding result writers."""

from __future__ import annotations

from pathlib import Path

from datasets import Dataset

from genslm_embeddings.embed.writers import HuggingFaceWriter


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
