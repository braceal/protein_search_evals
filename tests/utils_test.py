"""Tests for configuration and FASTA utilities."""

from __future__ import annotations

from pathlib import Path

from Bio.SeqRecord import SeqRecord

from genslm_embeddings.utils import BaseConfig
from genslm_embeddings.utils import Sequence
from genslm_embeddings.utils import batch_data
from genslm_embeddings.utils import chunk_fasta_file
from genslm_embeddings.utils import read_fasta
from genslm_embeddings.utils import write_fasta


class ExampleConfig(BaseConfig):
    """Small concrete configuration used to exercise serialization."""

    count: int
    labels: list[str]


def test_config_round_trips_json_and_yaml(tmp_path: Path) -> None:
    config = ExampleConfig(count=3, labels=['alpha', 'beta'])

    json_path = tmp_path / 'config.json'
    yaml_path = tmp_path / 'config.yaml'
    config.write_json(json_path)
    config.write_yaml(yaml_path)

    assert ExampleConfig.from_json(json_path) == config
    assert ExampleConfig.from_yaml(yaml_path) == config


def test_batch_data_preserves_order_and_remainder() -> None:
    assert batch_data(list(range(7)), 3) == [[0, 1, 2], [3, 4, 5], [6]]
    assert batch_data([], 3) == []


def test_fasta_read_write_and_append(tmp_path: Path) -> None:
    fasta_path = tmp_path / 'sequences.fasta'
    sequences = [
        Sequence(sequence='MKT', tag='protein-1 first protein'),
        Sequence(sequence='ACDE', tag='protein-2'),
    ]

    write_fasta(sequences[0], fasta_path)
    write_fasta(sequences[1], fasta_path, mode='a')

    assert read_fasta(fasta_path) == sequences


def test_chunk_fasta_file_applies_header_formatter(tmp_path: Path) -> None:
    input_path = tmp_path / 'input.fasta'
    output_dir = tmp_path / 'chunks'
    write_fasta(
        [
            Sequence(sequence='AAA', tag='one'),
            Sequence(sequence='CCC', tag='two'),
            Sequence(sequence='GGG', tag='three'),
        ],
        input_path,
    )

    def prefix_header(record: SeqRecord) -> SeqRecord:
        record.id = f'test-{record.id}'
        record.description = ''
        return record

    chunk_fasta_file(input_path, output_dir, 2, prefix_header)

    chunks = sorted(output_dir.glob('*.fasta'))
    assert [path.name for path in chunks] == [
        'input_0000.fasta',
        'input_0001.fasta',
    ]
    assert read_fasta(chunks[0]) == [
        Sequence(sequence='AAA', tag='test-one'),
        Sequence(sequence='CCC', tag='test-two'),
    ]
    assert read_fasta(chunks[1]) == [
        Sequence(sequence='GGG', tag='test-three'),
    ]
