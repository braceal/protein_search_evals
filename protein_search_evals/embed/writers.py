"""Hugging face dataset writer for saving results to disk."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import concatenate_datasets
from datasets import Dataset


class HuggingFaceWriter:
    """Hugging face writer for saving results to disk."""

    def write(self, output_dir: Path, result: dict[str, Any]) -> None:
        """Write the result to disk.

        Parameters
        ----------
        output_dir : Path
            The output directory to write the dataset to.
        result : dict[str, Any]
            The result dictionary containing the contents of the dataset
            (e.g., embeddings, sequences, metadata, etc).
        """
        # Create a dataset from the result
        dataset = Dataset.from_dict(result)

        # Write the dataset to disk
        dataset.save_to_disk(output_dir)

    def merge(self, dataset_dirs: list[Path], output_dir: Path) -> None:
        """Merge the datasets from multiple directories.

        Parameters
        ----------
        dataset_dirs : list[Path]
            The dataset directories to merge.
        output_dir : Path
            The output directory to write the merged dataset to.
        """
        # Load all the datasets
        all_datasets = [Dataset.load_from_disk(p) for p in dataset_dirs]

        # Concatenate the datasets
        dataset = concatenate_datasets(all_datasets)

        # Write the dataset to disk
        dataset.save_to_disk(output_dir)


class TokenEmbeddingWriter(HuggingFaceWriter):
    """Interface for writing token embeddings to disk."""

    def __init__(self, output_dir: Path, buffer_size: int = 50_000) -> None:
        """Initialize the token embedding writer.

        Parameters
        ----------
        output_dir : Path
            The output directory to write the token embedding subdirs to.
        buffer_size : int, optional
            The buffer size for writing token embeddings to disk,
            by default 50_000.
        """
        self.output_dir = output_dir
        self.buffer_size = buffer_size

        # Initialize the buffer for storing embeddings
        self.buffer: list[np.ndarray] = []

        # Create a file name counter for the buffer
        self.file_counter = 0
        if self.output_dir.exists():
            self.file_counter = len(list(self.output_dir.iterdir()))

    def _write(self) -> None:
        """Write the buffer to disk and reset the buffer."""
        # Create an output directory for the embeddings
        dir_name = f'token_embeddings_{self.file_counter:06d}'
        output_dir = self.output_dir / dir_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create the result dictionary
        result = {'embeddings': self.buffer}

        # Write the result to disk
        super().write(output_dir, result)

        # Reset the buffer and increment the file counter
        self.buffer = []
        self.file_counter += 1

    def flush(self) -> None:
        """Flush the buffer to disk."""
        if self.buffer:
            self._write()

    def append(
        self,
        embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> None:
        """Write the embeddings to disk.

        Parameters
        ----------
        embeddings : torch.Tensor
            The embeddings to write to disk.
        attention_mask : torch.Tensor
            The attention mask for the embeddings.
        """
        # Get the sequence lengths
        seq_lengths = attention_mask.sum(axis=1)

        # Make a list of ragged embeddings (no padding)
        ragged_embeddings = [
            emb[1 : seq_len - 1].cpu().numpy()
            for emb, seq_len in zip(embeddings, seq_lengths)
        ]

        # Extend the buffer with the ragged embeddings
        self.buffer.extend(ragged_embeddings)

        # Check if the buffer is full
        if len(self.buffer) >= self.buffer_size:
            # Write the buffer to disk
            self._write()
