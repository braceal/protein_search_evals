"""Module for storing token embeddings with HDF5."""

from __future__ import annotations

import weakref
from dataclasses import dataclass
from dataclasses import field
from functools import cached_property
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np

from protein_search_evals.timer import timeit_decorator


@dataclass
class TokenEmbedInfo:
    """Dataclass for storing token embeddings."""

    sequences: list[str] = field(
        default_factory=list,
        metadata={
            'description': 'The sequences corresponding to the embeddings',
        },
    )
    embeddings: list[np.ndarray] = field(
        default_factory=list,
        metadata={
            'description': 'The token embeddings, each of shape'
            ' (seq_len, hidden_size)',
        },
    )

    def __len__(self) -> int:
        """Return the number of embeddings."""
        return len(self.embeddings)

    @property
    def embedding_dim(self) -> int:
        """Return the embedding dimension."""
        if not self.embeddings:
            raise ValueError('No embeddings found in the buffer.')
        return self.embeddings[0].shape[-1]

    @property
    def seq_lengths(self) -> list[int]:
        """Return the sequence lengths."""
        return [len(emb) for emb in self.embeddings]


class HDF5TokenEmbeddings:
    """Interface for reading/writing token embeddings with HDF5."""

    def __init__(
        self,
        file: str | Path,
        buffer_size: int = 50_000,
        max_sequence_length: int = 2048,
    ) -> None:
        """Initialize the token embedding container.

        Parameters
        ----------
        file : str | Path
            The HDF5 file backing for the token embeddings.
        buffer_size : int, optional
            The buffer size for writing token embeddings to disk,
            (only used for writing, will flush when buffer is full),
            by default 50_000.
        max_sequence_length : int, optional
            The maximum sequence length for the embeddings (only
            used for writing), by default 2048.
        """
        self.file = Path(file)
        self.buffer_size = buffer_size
        self.max_sequence_length = max_sequence_length

        # Initialize the buffer for storing embeddings and sequence lengths
        self.buffer = TokenEmbedInfo()

        # Keep an open file handle for reading
        self._file_handle = None

        # Ensure the file is closed when the object is deleted
        # NOTE: This is needed since if the logic from self.close is
        # called in the destructor, the python GC may have already
        # destroyed internal objects that the HDF5 file depends on
        # which causes an error when closing the file.
        self._finalizer = weakref.finalize(self, self.close)

    def _append_batch_to_hdf5(self, buffer: TokenEmbedInfo) -> None:
        # Get the batch size, embeddings, and embedding dimension
        batch_size = len(buffer)
        batch = buffer.embeddings
        embedding_dim = buffer.embedding_dim
        lengths = np.array(buffer.seq_lengths, dtype=np.int32)
        sequences = np.array(buffer.sequences)

        with h5py.File(self.file, 'a') as f:
            # Create datasets on the first batch
            if 'token_embeddings' not in f:
                f.create_dataset(
                    'token_embeddings',
                    shape=(0, self.max_sequence_length, embedding_dim),
                    maxshape=(None, self.max_sequence_length, embedding_dim),
                    dtype=np.float16,
                    chunks=(1, self.max_sequence_length, embedding_dim),
                    compression='lzf',
                )
                f.create_dataset(
                    'sequence_lengths',
                    shape=(0,),
                    maxshape=(None,),
                    dtype=np.int32,
                    chunks=(batch_size,),
                )
                f.create_dataset(
                    'sequences',
                    shape=(0,),
                    maxshape=(None,),
                    dtype=h5py.string_dtype(encoding='utf-8'),
                    chunks=(batch_size,),
                )

            # Get the datasets
            embeddings_dset = f['token_embeddings']
            lengths_dset = f['sequence_lengths']
            sequence_dset = f['sequences']

            # Current number of samples in the dataset
            current_size = embeddings_dset.shape[0]

            # Resize datasets to fit the new batch
            new_size = current_size + batch_size
            embeddings_dset.resize(
                (new_size, self.max_sequence_length, embedding_dim),
            )
            lengths_dset.resize((new_size,))
            sequence_dset.resize((new_size,))

            # Prepare padded batch
            padded_batch = np.zeros(
                (batch_size, self.max_sequence_length, embedding_dim),
                dtype=np.float16,
            )
            for i, embedding in enumerate(batch):
                padded_batch[i, : embedding.shape[0], :] = embedding

            # Write to the datasets
            embeddings_dset[current_size:new_size] = padded_batch
            lengths_dset[current_size:new_size] = lengths
            sequence_dset[current_size:new_size] = sequences

    @timeit_decorator('hdf5-token-embedddings')
    def flush(self) -> None:
        """Flush the buffer to disk in the background."""
        # If the buffer is empty, return
        if len(self.buffer) == 0:
            return

        # Swap out the buffer so we can continue collecting embeddings
        buffer_to_flush = self.buffer
        self.buffer = TokenEmbedInfo()

        self._append_batch_to_hdf5(buffer_to_flush)

    def append(
        self,
        sequences: list[str],
        embeddings: list[np.ndarray],
    ) -> None:
        """Write the embeddings to disk.

        Will flush the buffer if the buffer size is reached.

        Parameters
        ----------
        embeddings : list[np.ndarray]
            The token embeddings stored as a list of ragged arrays
            (batch_size, *seq_length, hidden_dim) to write to disk.
        """
        # Extend the buffer with the sequences and ragged token embeddings
        self.buffer.sequences.extend(sequences)
        self.buffer.embeddings.extend(embeddings)

        # Check if the buffer is full
        if len(self.buffer) >= self.buffer_size:
            # Write the buffer to disk
            self.flush()

    def _get_file_handle(self) -> h5py.File:
        """Get the file handle for reading the token embeddings."""
        if self._file_handle is None:
            self._file_handle = h5py.File(self.file, 'r')

        return self._file_handle

    def close(self) -> None:
        """Close the file handle when the object is deleted."""
        if self._file_handle is not None:
            self._file_handle.close()
            self._file_handle = None

    def __len__(self) -> int:
        """Return the number of token embeddings in the dataset."""
        file_handle = self._get_file_handle()
        return file_handle['token_embeddings'].shape[0]

    def __getitem__(
        self,
        idx: int | slice | Sequence[int] | list[int],
    ) -> np.ndarray:
        """Retrieve one or more token embeddings.

        Parameters
        ----------
        idx : int | slice | Sequence[int]
            The index or indices to retrieve.

        Returns
        -------
        np.ndarray
            The token embeddings for the given index or indices.
            They are trimmed based on the sequence lengths (no padding)
            which makes it a ragged array.

        Raises
        ------
        TypeError
            If the index type is not supported.
        """
        file_handle = self._get_file_handle()
        embeddings = file_handle['token_embeddings']
        lengths = file_handle['sequence_lengths']

        if isinstance(idx, int):
            # Single index
            return embeddings[idx, : lengths[idx], :]

        elif isinstance(idx, (slice, list, np.ndarray)):
            # Slice or list/array of indices
            # Trim padding based on lengths
            token_embeddings = [
                emb[:length]
                for emb, length in zip(embeddings[idx], lengths[idx])
            ]
            return np.array(token_embeddings, dtype=object)
        else:
            raise TypeError(
                f'Indexing with type {type(idx)} is not supported.',
            )

    @cached_property
    def sequence_to_index(self) -> dict[str, int]:
        """Compute a cached mapping from sequence to index in the dataset.

        Returns
        -------
        dict[str, int]
            The mapping from sequence to index in the dataset.
        """
        file_handle = self._get_file_handle()
        sequences = file_handle['sequences']
        return {str(seq): idx for idx, seq in enumerate(sequences)}

    def get_embeddings(self, sequences: list[str]) -> list[np.ndarray]:
        """Get the embeddings for the given sequences.

        Parameters
        ----------
        sequences : list[str]
            The sequences to get the embeddings for.

        Returns
        -------
        np.ndarray
            The embeddings for the given sequences.

        Raises
        ------
        KeyError
            If a sequence is not found in the dataset.
        """
        # Get the indices for the sequences
        indices = [self.sequence_to_index[seq] for seq in sequences]

        return self[indices]
