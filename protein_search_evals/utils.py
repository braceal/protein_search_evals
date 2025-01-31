"""Utility functions for reading and writing fasta files."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from typing import TypeVar

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel

T = TypeVar('T')


class BaseConfig(BaseModel):
    """An interface to add JSON/YAML serialization to Pydantic models."""

    # A name literal to correctly identify and construct nested models
    # which have many possible options.
    name: Literal[''] = ''

    def write_json(self, path: str | Path) -> None:
        """Write the model to a JSON file.

        Parameters
        ----------
        path : str | Path
            The path to the JSON file.
        """
        with open(path, 'w') as fp:
            json.dump(self.model_dump(), fp, indent=2)

    @classmethod
    def from_json(cls: type[T], path: str | Path) -> T:
        """Load the model from a JSON file.

        Parameters
        ----------
        path : str | Path
            The path to the JSON file.

        Returns
        -------
        T
            A specific BaseConfig instance.
        """
        with open(path) as fp:
            data = json.load(fp)
        return cls(**data)

    def write_yaml(self, path: str | Path) -> None:
        """Write the model to a YAML file.

        Parameters
        ----------
        path : str | Path
            The path to the YAML file.
        """
        with open(path, 'w') as fp:
            yaml.dump(
                json.loads(self.model_dump_json()),
                fp,
                indent=4,
                sort_keys=False,
            )

    @classmethod
    def from_yaml(cls: type[T], path: str | Path) -> T:
        """Load the model from a YAML file.

        Parameters
        ----------
        path : str | Path
            The path to the YAML file.

        Returns
        -------
        T
            A specific BaseConfig instance.
        """
        with open(path) as fp:
            raw_data = yaml.safe_load(fp)
        return cls(**raw_data)


def batch_data(data: list[T], chunk_size: int) -> list[list[T]]:
    """Batch data into chunks of size chunk_size.

    Parameters
    ----------
    data : list[T]
        The data to batch.
    chunk_size : int
        The size of each batch.

    Returns
    -------
    list[list[T]]
        The batched data.
    """
    batches = [
        data[i * chunk_size : (i + 1) * chunk_size]
        for i in range(0, len(data) // chunk_size)
    ]
    if len(data) > chunk_size * len(batches):
        batches.append(data[len(batches) * chunk_size :])
    return batches


@dataclass
class Sequence:
    """Biological sequence dataclass."""

    sequence: str
    """Biological sequence (Nucleotide/Amino acid sequence)."""
    tag: str
    """Sequence description tag."""


def read_fasta(fasta_file: str | Path) -> list[Sequence]:
    """Read fasta file sequences and description tags into dataclass."""
    text = Path(fasta_file).read_text()
    pattern = re.compile('^>', re.MULTILINE)
    non_parsed_seqs = re.split(pattern, text)[1:]
    lines = [
        line.replace('\n', '')
        for seq in non_parsed_seqs
        for line in seq.split('\n', 1)
    ]

    return [
        Sequence(sequence=seq, tag=tag)
        for seq, tag in zip(lines[1::2], lines[::2])
    ]


def write_fasta(
    sequences: Sequence | list[Sequence],
    fasta_file: str | Path,
    mode: str = 'w',
) -> None:
    """Write or append sequences to a fasta file."""
    seqs = [sequences] if isinstance(sequences, Sequence) else sequences
    with open(fasta_file, mode) as f:
        for seq in seqs:
            f.write(f'>{seq.tag}\n{seq.sequence}\n')
