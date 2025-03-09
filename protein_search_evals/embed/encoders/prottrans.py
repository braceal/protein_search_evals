"""Encoder for the ProtTrans model."""

from __future__ import annotations

from typing import Any
from typing import Literal

import torch
from pydantic import Field
from transformers import BatchEncoding
from transformers import PreTrainedTokenizer
from transformers import T5EncoderModel
from transformers import T5Tokenizer

from protein_search_evals.embed.encoders.base import Encoder
from protein_search_evals.embed.encoders.base import EncoderConfig


class ProtTransEncoderConfig(EncoderConfig):
    """Config for the ProtTrans encoder."""

    # The name of the encoder
    name: Literal['prottrans'] = 'prottrans'

    pretrained_model_name_or_path: str = Field(
        default='Rostlab/prot_t5_xl_half_uniref50-enc',
        description='The model id.',
    )
    tokenizer_path: str | None = Field(
        default=None,
        description='The model tokenizer path (if different from the model).',
    )
    half_precision: bool = Field(
        default=True,
        description='Whether to use half precision for the model.',
    )


class ProtTransEncoder(Encoder):
    """Encoder for the ProtTrans model.

    For more information, see the ProtTrans paper:
    "ProtTrans: Toward Understanding the Language of Life Through
    Self-Supervised Learning", Elnaggar et al. (2021).
    https://ieeexplore.ieee.org/document/9477085
    """

    def __init__(
        self,
        pretrained_model_name_or_path: str,
        tokenizer_path: str | None = None,
        half_precision: bool = True,
        **kwargs: Any,
    ):
        """Initialize the ProtTrans encoder.

        Parameters
        ----------
        pretrained_model_name_or_path : str
            The model id.
        tokenizer_path : str | None, optional
            The model tokenizer path (if different from the model),
            by default None.
        half_precision : bool, optional
            Whether to use half precision for the model, by default True.
        **kwargs : Any
            Additional base arguments, see `Encoder`.
        """
        # Initialize the base encoder
        super().__init__(**kwargs)

        # Load the model
        model = T5EncoderModel.from_pretrained(pretrained_model_name_or_path)

        # Load the tokenizer
        tokenizer = T5Tokenizer.from_pretrained(
            tokenizer_path or pretrained_model_name_or_path,
            do_lower_case=False,
        )

        # Set the model max length for proper truncation
        tokenizer.model_max_length = model.config.n_positions

        # Convert the model to half precision
        if half_precision:
            model.half()

        # Set the model to evaluation mode
        model.eval()

        # Load the model onto the device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        # Set persistent attributes
        self.model = model
        self._tokenizer = tokenizer

    @property
    def dtype(self) -> torch.dtype:
        """Get the data type of the encoder."""
        return self.model.dtype

    @property
    def device(self) -> torch.device:
        """Get the device of the encoder."""
        return self.model.device

    @property
    def max_length(self) -> int:
        """Get the maximum sequence length of the encoder."""
        return self.model.config.n_positions

    @property
    def embedding_size(self) -> int:
        """Get the embedding size of the encoder."""
        # It's 512 for Rostlab/prot_t5_xl_half_uniref50-enc
        return self.model.config.d_model

    @property
    def tokenizer(self) -> PreTrainedTokenizer:
        """Get the tokenizer of the encoder."""
        return self._tokenizer

    def encode(self, batch_encoding: BatchEncoding) -> torch.Tensor:
        """Encode the sequence.

        Parameters
        ----------
        batch_encoding : BatchEncoding
            The batch encoding of the sequence (containing the input_ids,
            attention_mask, and token_type_ids).

        Returns
        -------
        torch.Tensor
            The embeddings of the sequence extracted from the last hidden state
            (shape: [num_sequences, sequence_length, embedding_size])
        """
        # Get the model outputs with a forward pass
        outputs = self.model(**batch_encoding)

        return outputs.last_hidden_state[-1]
