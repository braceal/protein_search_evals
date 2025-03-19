"""Encoder interface for all encoders to follow."""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from pydantic import BaseModel
from pydantic import Field
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import BatchEncoding
from transformers import PreTrainedTokenizer

from protein_search_evals.embed.embeddings import HDF5TokenEmbeddings
from protein_search_evals.embed.poolers import average_pool


class InMemoryDataset(Dataset):
    """Holds the data in memory for efficient batching."""

    def __init__(self, data: list[str]) -> None:
        """Initialize the dataset.

        Parameters
        ----------
        data : list[str]
            The list of string sequences to enumerate.
        """
        self.data = data

    def __len__(self) -> int:
        """Length of the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> str:
        """Get an item from the dataset."""
        return self.data[idx]


class DataCollator:
    """Data collator for batching sequences."""

    def __init__(self, tokenizer: PreTrainedTokenizer) -> None:
        """Initialize the data collator."""
        self.tokenizer = tokenizer

    def __call__(self, batch: list[str]) -> BatchEncoding:
        """Collate the batch of sequences."""
        return self.tokenizer(
            batch,
            padding=True,
            truncation=True,
            return_tensors='pt',
        )


class EncoderConfig(BaseModel):
    """Configuration for the Encoder."""

    normalize_pooled_embeddings: bool = Field(
        default=False,
        description='Whether to normalize the pooled embeddings.',
    )
    dataloader_pin_memory: bool = Field(
        default=True,
        description='Whether to pin memory for the dataloader.',
    )
    dataloader_batch_size: int = Field(
        default=8,
        description='The batch size for inference.',
    )
    dataloader_num_data_workers: int = Field(
        default=4,
        description='Number of data workers for batching.',
    )
    cached_token_embeddings_path: str | Path | None = Field(
        default=None,
        description='Path to the cached HDF5 token embeddings.',
    )


@dataclass
class EncoderOutput:
    """Output of the encoder."""

    pool_embeddings: np.ndarray = field(
        metadata={'description': 'Pooled embeddings.'},
    )
    token_embeddings: list[np.ndarray] | None = field(
        default=None,
        metadata={'description': 'Token embeddings stored as a ragged array.'},
    )


class Encoder(ABC):
    """Encoder protocol for all encoders to follow."""

    def __init__(
        self,
        normalize_pooled_embeddings: bool = False,
        dataloader_pin_memory: bool = True,
        dataloader_batch_size: int = 8,
        dataloader_num_data_workers: int = 4,
        cached_token_embeddings_path: str | Path | None = None,
        verbose: bool = False,
    ) -> None:
        """Initialize the encoder.

        Parameters
        ----------
        normalize_pooled_embeddings : bool, optional
            Whether to normalize the pooled embeddings, by default False.
        dataloader_pin_memory : bool, optional
            Whether to pin memory for the dataloader, by default True.
        dataloader_batch_size : int, optional
            The batch size for inference, by default 8.
        dataloader_num_data_workers : int, optional
            Number of data workers for batching, by default 4.
        cached_token_embeddings_path : str | Path | None, optional
            Path to the cached HDF5 token embeddings, by default None.
        verbose : bool, optional
            Whether to print verbose output (progress bar for embedding
            computation), by default False.
        """
        self.normalize_pooled_embeddings = normalize_pooled_embeddings
        self.dataloader_pin_memory = dataloader_pin_memory
        self.dataloader_batch_size = dataloader_batch_size
        self.dataloader_num_data_workers = dataloader_num_data_workers
        self.cached_token_embeddings_path = cached_token_embeddings_path
        self.verbose = verbose

        # Load the cached token embeddings
        if self.cached_token_embeddings_path is not None:
            self.token_embedding_reader = HDF5TokenEmbeddings(
                file=self.cached_token_embeddings_path,
            )

    @property
    def sos_token(self) -> bool:
        """Whether the encoder has a start token.

        Override this property if the encoder does not have a start token.
        """
        return True

    @property
    def eos_token(self) -> bool:
        """Whether the encoder has an end token.

        Override this property if the encoder does not have an end token.
        """
        return True

    @property
    @abstractmethod
    def dtype(self) -> torch.dtype:
        """Get the data type of the encoder."""
        ...

    @property
    @abstractmethod
    def device(self) -> torch.device:
        """Get the device of the encoder."""
        ...

    @property
    @abstractmethod
    def max_length(self) -> int:
        """Get the maximum sequence length of the encoder."""
        ...

    @property
    @abstractmethod
    def embedding_size(self) -> int:
        """Get the embedding size of the encoder."""
        ...

    @property
    @abstractmethod
    def tokenizer(self) -> PreTrainedTokenizer:
        """Get the tokenizer of the encoder."""
        ...

    @abstractmethod
    def encode(self, batch_encoding: BatchEncoding) -> torch.Tensor:
        """Encode the sequence.

        Parameters
        ----------
        batch_encoding : BatchEncoding
            The batch encoding of the sequence.

        Returns
        -------
        torch.Tensor
            The embeddings of the sequence
            (shape: [num_sequences, sequence_length, embedding_size])
        """
        ...

    def pool(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Pool the hidden states.

        By default, we use average pooling. Override this method for
        custom pooling operations.

        Parameters
        ----------
        hidden_states : torch.Tensor
            The hidden states of the sequence
            (shape: [num_sequences, sequence_length, hidden_size])
        attention_mask : torch.Tensor
            The attention mask of the sequence
            (shape: [num_sequences, sequence_length])

        Returns
        -------
        torch.Tensor
            The pooled hidden states
            (shape: [num_sequences, hidden_size])
        """
        return average_pool(
            embeddings=hidden_states,
            attention_mask=attention_mask,
            sos_token=self.sos_token,
            eos_token=self.eos_token,
        )

    def get_dataloader(self, sequences: list[str]) -> DataLoader:
        """Instantiate a dataloader for the sequences.

        Override this method for custom dataloaders.

        Parameters
        ----------
        sequences : list[str]
            The list of sequences to encode.

        Returns
        -------
        DataLoader
            The dataloader instance.
        """
        # Capitalize the sequences
        data = [x.upper() for x in sequences]

        # Instantiate the dataloader
        return DataLoader(
            pin_memory=self.dataloader_pin_memory,
            batch_size=self.dataloader_batch_size,
            num_workers=self.dataloader_num_data_workers,
            dataset=InMemoryDataset(data),
            collate_fn=DataCollator(self.tokenizer),
        )

    def _embeddings_from_cache(
        self,
        sequences: list[str],
        normalize_pooled_embeddings: bool,
        return_token_embeddings: bool = False,
    ) -> EncoderOutput | None:
        """Get embeddings from the cache.

        Parameters
        ----------
        sequences : list[str]
            The list of sequences to embed.
        normalize_pooled_embeddings : bool
            Whether to normalize the pooled embeddings.
        return_token_embeddings : bool, optional
            Whether to return the token embeddings, by default False.

        Returns
        -------
        EncoderOutput | None
            The pooled embeddings and token embeddings,
            or None if not found in cache.
        """
        try:
            # Get the token embeddings from the cache
            token_embeddings = self.token_embedding_reader.get_embeddings(
                sequences,
            )
        except KeyError:
            # If a sequence is not found, compute the embeddings
            return None

        # Compute attention masks (including space for special tokens)
        special_token_offset = int(self.sos_token) + int(self.eos_token)
        attention_masks = [
            torch.ones(len(seq) + special_token_offset, dtype=torch.long)
            for seq in token_embeddings
        ]

        # Compute the pooled embeddings
        pooled_embeds = torch.tensor(
            [
                self.pool(torch.tensor(emb, dtype=self.dtype), mask)
                for emb, mask in zip(token_embeddings, attention_masks)
            ],
        )

        # Normalize the embeddings
        if normalize_pooled_embeddings:
            pooled_embeds = F.normalize(pooled_embeds, p=2, dim=-1)

        # Return the embeddings
        if return_token_embeddings:
            return EncoderOutput(
                pool_embeddings=pooled_embeds.numpy(),
                token_embeddings=token_embeddings,
            )
        else:
            return EncoderOutput(
                pool_embeddings=pooled_embeds.numpy(),
                token_embeddings=None,
            )

    @torch.no_grad()
    def compute_embeddings(
        self,
        sequences: list[str],
        normalize_pooled_embeddings: bool | None = None,
        token_embedding_writer: HDF5TokenEmbeddings | None = None,
        return_token_embeddings: bool = False,
    ) -> EncoderOutput:
        """Compute hidden embeddings.

        Parameters
        ----------
        sequences : list[str]
            A list of sequences to embed.
        normalize_pooled_embeddings : bool, optional
            Whether to normalize the pooled embeddings, by default None
            will use the instance attribute defined at initialization.
        token_embedding_writer : HDF5TokenEmbeddings, optional
            A writer for dense embeddings, by default None.
        return_token_embeddings : bool, optional
            Whether to return the token embeddings, by default False.

        Returns
        -------
        EncoderOutput
            The pooled embeddings and token embeddings.
        """
        # Decide whether to normalize the embeddings
        if normalize_pooled_embeddings is None:
            normalize_pooled_embeddings = self.normalize_pooled_embeddings

        # If a cached token embedding reader is provided, use it
        if self.cached_token_embeddings_path is not None:
            cached_output = self._embeddings_from_cache(
                sequences,
                normalize_pooled_embeddings,
                return_token_embeddings=return_token_embeddings,
            )
            # If the embeddings are found in the cache, return them
            # otherwise, we compute embeddings as usual
            if cached_output is not None:
                return cached_output

        # Create a dataloader for the sequences
        dataloader = self.get_dataloader(sequences)

        # Initialize a torch tensor for storing embeddings in host memory
        all_embeddings = torch.empty(
            (len(sequences), self.embedding_size),
            dtype=self.dtype,
        )
        token_embeddings = []

        # Index for storing embeddings
        idx = 0

        for batch in tqdm(
            dataloader,
            desc='Computing embeddings',
            disable=not self.verbose,
        ):
            # Move the batch to the model device
            inputs = batch.to(self.device)

            # Get the model outputs with a forward pass
            embeds = self.encode(inputs)

            # Compute the pooled embeddings
            pooled_embeds = self.pool(embeds, inputs.attention_mask)

            # Normalize the embeddings
            if normalize_pooled_embeddings:
                pooled_embeds = F.normalize(pooled_embeds, p=2, dim=-1)

            # Get the batch size
            batch_size = inputs.attention_mask.shape[0]

            # Store the pooled embeddings in the output buffer
            all_embeddings[idx : idx + batch_size, :] = pooled_embeds.cpu()

            # If the token embeddings are requested, prepare a ragged list
            if return_token_embeddings or token_embedding_writer is not None:
                # Get the sequence lengths
                seq_lengths = inputs.attention_mask.sum(axis=1)

                # Make a list of ragged embeddings (no padding)
                start, stop = int(self.sos_token), int(self.eos_token)
                ragged_embeddings = [
                    emb[start : seq_len - stop].cpu().numpy()
                    for emb, seq_len in zip(embeds, seq_lengths)
                ]

            # If the token embeddings are requested, append to the list
            if return_token_embeddings:
                token_embeddings.extend(ragged_embeddings)

            # If a dense writer is provided, write the embeddings
            if token_embedding_writer is not None:
                # Get the sequences associated with the embeddings
                seqs = sequences[idx : idx + batch_size]
                # Write the embeddings to disk
                token_embedding_writer.append(seqs, ragged_embeddings)

            # Increment the output buffer index by the batch size
            idx += batch_size

        # Construct the encoder result
        return EncoderOutput(
            pool_embeddings=all_embeddings.numpy(),
            token_embeddings=token_embeddings if token_embeddings else None,
        )
