"""Encoder interface for all encoders to follow."""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod

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

    normalize_embeddings: bool = Field(
        default=False,
        description='Whether to normalize the embeddings.',
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


class Encoder(ABC):
    """Encoder protocol for all encoders to follow."""

    def __init__(
        self,
        normalize_embeddings: bool = False,
        dataloader_pin_memory: bool = True,
        dataloader_batch_size: int = 8,
        dataloader_num_data_workers: int = 4,
    ) -> None:
        """Initialize the encoder.

        Parameters
        ----------
        normalize_embeddings : bool, optional
            Whether to normalize the embeddings, by default False.
        dataloader_pin_memory : bool, optional
            Whether to pin memory for the dataloader, by default True.
        dataloader_batch_size : int, optional
            The batch size for inference, by default 8.
        dataloader_num_data_workers : int, optional
            Number of data workers for batching, by default 4.
        """
        self.normalize_embeddings = normalize_embeddings
        self.dataloader_pin_memory = dataloader_pin_memory
        self.dataloader_batch_size = dataloader_batch_size
        self.dataloader_num_data_workers = dataloader_num_data_workers

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
        return average_pool(hidden_states, attention_mask)

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

    @torch.no_grad()
    def compute_pooled_embeddings(
        self,
        sequences: list[str],
        normalize_embeddings: bool | None = None,
    ) -> np.ndarray:
        """Compute pooled hidden embeddings.

        Parameters
        ----------
        sequences : list[str]
            A list of sequences to embed.
        normalize : bool, optional
            Whether to normalize the embeddings, by default None
            will use the instance attribute defined at initialization.

        Returns
        -------
        np.ndarray
            A numpy array of pooled hidden embeddings.
        """
        # Decide whether to normalize the embeddings
        if normalize_embeddings is None:
            normalize_embeddings = self.normalize_embeddings

        # Create a dataloader for the sequences
        dataloader = self.get_dataloader(sequences)

        # Initialize a torch tensor for storing embeddings in host memory
        all_embeddings = torch.empty(
            (len(sequences), self.embedding_size),
            dtype=self.dtype,
        )

        # Index for storing embeddings
        idx = 0

        for batch in tqdm(dataloader, desc='Computing pooled embeddings'):
            # Move the batch to the model device
            inputs = batch.to(self.device)

            # Get the model outputs with a forward pass
            embeddings = self.encode(inputs)

            # Compute the pooled embeddings
            pooled_embeds = self.pool(embeddings, inputs.attention_mask)

            # Normalize the embeddings
            if normalize_embeddings:
                pooled_embeds = F.normalize(pooled_embeds, p=2, dim=-1)

            # Get the batch size
            batch_size = inputs.attention_mask.shape[0]

            # Store the pooled embeddings in the output buffer
            all_embeddings[idx : idx + batch_size, :] = pooled_embeds.cpu()

            # Increment the output buffer index by the batch size
            idx += batch_size

        return all_embeddings.numpy()

    # TODO: Function to compute non-pooled embeddings
