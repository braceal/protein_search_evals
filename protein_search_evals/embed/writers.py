"""Hugging face dataset writer for saving results to disk."""

from __future__ import annotations

from pathlib import Path
from typing import Any

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
