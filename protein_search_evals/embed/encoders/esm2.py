"""Encoder for the ESM-2 model."""

from __future__ import annotations

import warnings
from typing import Literal

import torch
from pydantic import Field
from transformers import BatchEncoding
from transformers import EsmTokenizer
from transformers import PreTrainedTokenizer

from protein_search_evals.embed.encoders.base import Encoder
from protein_search_evals.embed.encoders.base import EncoderConfig


class Esm2EncoderConfig(EncoderConfig):
    """Config for the ESM-2 encoder."""

    # The name of the encoder
    name: Literal['esm2'] = 'esm2'

    pretrained_model_name_or_path: str = Field(
        default='facebook/esm2_t6_8M_UR50D',
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
    enable_faesm: bool = Field(
        default=False,
        description='Whether to use the faesm implementation (faster).',
    )


class Esm2Encoder(Encoder):
    """Encoder for the ESM-2 model."""

    def __init__(
        self,
        pretrained_model_name_or_path: str,
        tokenizer_path: str | None = None,
        half_precision: bool = True,
        enable_faesm: bool = False,
    ):
        """Initialize the ESM-2 encoder.

        Parameters
        ----------
        pretrained_model_name_or_path : str
            The model id.
        tokenizer_path : str | None, optional
            The model tokenizer path (if different from the model),
            by default None.
        half_precision : bool, optional
            Whether to use half precision for the model, by default True.
        enable_faesm : bool, optional
            Whether to use the faesm implementation (faster), by default False.
        """
        # Check if faesm is enabled
        if enable_faesm:
            try:
                from faesm.esm import FAEsmForMaskedLM as EsmForMaskedLM

                print('Using faesm implementation.')
            except ImportError:
                warnings.warn(
                    'faesm is not installed. Falling back to transformers.',
                    stacklevel=2,
                )
                from transformers import EsmForMaskedLM
        else:
            from transformers import EsmForMaskedLM

        # Load model and tokenizer
        model = EsmForMaskedLM.from_pretrained(pretrained_model_name_or_path)

        # Load the tokenizer
        if tokenizer_path is None:
            tokenizer_path = pretrained_model_name_or_path
        tokenizer = EsmTokenizer.from_pretrained(tokenizer_path)

        # Set the model max length for proper truncation
        tokenizer.model_max_length = model.config.max_position_embeddings

        # Convert the model to half precision
        if half_precision:
            model.half()

        # Set the model to evaluation mode
        model.eval()

        # Load the model onto the device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        # Set persistent attributes
        self.enable_faesm = enable_faesm
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
    def embedding_size(self) -> int:
        """Get the embedding size of the encoder."""
        return self.model.config.hidden_size

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
        outputs = self.model(
            **batch_encoding,
            output_hidden_states=not self.enable_faesm,
        )

        # Return the last hidden state
        if self.enable_faesm:
            return outputs['last_hidden_state']

        return outputs.hidden_states[-1]
