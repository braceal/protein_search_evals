"""Hugging face dataset writer for saving results to disk."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from typing import Mapping

import numpy as np
from datasets import concatenate_datasets
from datasets import Dataset

EMBEDDING_METADATA_FILE = 'embedding_metadata.json'


def select_embedding_layer(
    embeddings: np.ndarray,
    embedding_layer: int | None,
    metadata: Mapping[str, object] | None,
) -> np.ndarray:
    """Select a transformer block from stored embeddings."""
    embeddings = np.asarray(embeddings)
    if metadata is None:
        if embedding_layer is not None:
            raise ValueError(
                'embedding_layer cannot be used with final-layer-only '
                'embeddings',
            )
        return embeddings

    if embedding_layer is None:
        raise ValueError(
            'embedding_layer is required for multi-layer embeddings',
        )

    raw_layer_indices = metadata.get('layer_indices')
    if not isinstance(raw_layer_indices, list) or not all(
        isinstance(layer, int) for layer in raw_layer_indices
    ):
        raise ValueError('embedding metadata has invalid layer_indices')
    layer_indices = list(map(int, raw_layer_indices))
    if embedding_layer not in layer_indices:
        raise ValueError(
            f'embedding layer {embedding_layer} is not available; '
            f'stored layers are {layer_indices}',
        )

    layer_position = layer_indices.index(embedding_layer)
    if embeddings.ndim < 2 or embeddings.shape[-2] != len(layer_indices):
        raise ValueError(
            'multi-layer embedding shape does not match layer metadata; '
            f'got {embeddings.shape} for {len(layer_indices)} layers',
        )
    return np.take(embeddings, layer_position, axis=-2)


def load_embedding_metadata(dataset_dir: Path) -> dict[str, Any] | None:
    """Load optional embedding metadata stored beside a dataset."""
    metadata_path = dataset_dir / EMBEDDING_METADATA_FILE
    if not metadata_path.exists():
        return None
    metadata = json.loads(metadata_path.read_text())
    if not isinstance(metadata, dict):
        raise ValueError('embedding metadata must be a JSON object')
    return metadata


def write_embedding_metadata(
    dataset_dir: Path,
    metadata: dict[str, Any],
) -> None:
    """Write embedding metadata beside a saved dataset."""
    metadata_path = dataset_dir / EMBEDDING_METADATA_FILE
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))


def _validate_multilayer_shape(
    embeddings: np.ndarray,
    metadata: Mapping[str, object],
) -> None:
    """Validate a multi-layer embedding shape against its metadata."""
    layer_indices = metadata.get('layer_indices')
    if not isinstance(layer_indices, list) or not all(
        isinstance(layer, int) for layer in layer_indices
    ):
        raise ValueError('embedding metadata has invalid layer_indices')
    if embeddings.ndim != 3 or embeddings.shape[1] != len(layer_indices):
        raise ValueError(
            'multi-layer embedding shape does not match layer metadata',
        )
    hidden_dimension = metadata.get('hidden_dimension')
    if (
        hidden_dimension is not None
        and embeddings.shape[2] != hidden_dimension
    ):
        raise ValueError(
            'embedding hidden dimension does not match layer metadata',
        )


class HuggingFaceWriter:
    """Hugging face writer for saving results to disk."""

    def write(
        self,
        output_dir: Path,
        result: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Write the result to disk.

        Parameters
        ----------
        output_dir : Path
            The output directory to write the dataset to.
        result : dict[str, Any]
            The result dictionary containing the contents of the dataset
            (e.g., embeddings, sequences, metadata, etc).
        metadata : dict[str, Any] | None, optional
            Dataset-level embedding metadata, by default None.
        """
        if metadata is not None:
            _validate_multilayer_shape(
                np.asarray(result['embeddings']),
                metadata,
            )

        # Create a dataset from the result
        dataset = Dataset.from_dict(result)

        # Write the dataset to disk
        dataset.save_to_disk(output_dir)

        if metadata is not None:
            write_embedding_metadata(output_dir, metadata)

    def merge(self, dataset_dirs: list[Path], output_dir: Path) -> None:
        """Merge the datasets from multiple directories.

        Parameters
        ----------
        dataset_dirs : list[Path]
            The dataset directories to merge.
        output_dir : Path
            The output directory to write the merged dataset to.
        """
        # Validate optional metadata before loading and concatenating shards.
        all_metadata = [load_embedding_metadata(p) for p in dataset_dirs]
        metadata = all_metadata[0] if all_metadata else None
        if any(item != metadata for item in all_metadata[1:]):
            raise ValueError(
                'Cannot merge embedding datasets with different metadata',
            )

        # Load all the datasets
        all_datasets = [Dataset.load_from_disk(p) for p in dataset_dirs]

        if all_datasets and any(
            dataset.features != all_datasets[0].features
            for dataset in all_datasets[1:]
        ):
            raise ValueError(
                'Cannot merge embedding datasets with different schemas',
            )

        embedding_shapes = {
            tuple(np.asarray(dataset[0]['embeddings']).shape)
            for dataset in all_datasets
            if len(dataset) and 'embeddings' in dataset.column_names
        }
        if len(embedding_shapes) > 1:
            raise ValueError(
                'Cannot merge embedding datasets with different shapes',
            )

        # Concatenate the datasets
        dataset = concatenate_datasets(all_datasets)

        # Write the dataset to disk
        dataset.save_to_disk(output_dir)

        if metadata is not None:
            write_embedding_metadata(output_dir, metadata)
