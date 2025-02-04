"""Encoder for the ESM-Cambrian models."""

from __future__ import annotations

from typing import Any
from typing import Literal

import torch
from pydantic import Field
from pydantic import model_validator
from transformers import BatchEncoding
from transformers import PreTrainedTokenizer
from typing_extensions import Self

from protein_search_evals.embed.encoders.base import Encoder
from protein_search_evals.embed.encoders.base import EncoderConfig


# ESM-Cambrian doesn't provide the embedding size in the config
# so we need to set it manually based on the model name
def _validate_embedding_size(
    pretrained_model_name_or_path: str,
    embedding_size: int | None,
) -> int:
    # If the embedding size is explicitly set, return the value
    if isinstance(embedding_size, int):
        return embedding_size

    # Assert that the embedding size is None
    assert embedding_size is None

    # The embedding size based on the model name
    embedding_sizes = {
        'EvolutionaryScale/esmc-300m-2024-12': 960,
        'EvolutionaryScale/esmc-600m-2024-12': 1152,
    }

    # If the embedding size is not found, raise an error
    if pretrained_model_name_or_path not in embedding_sizes:
        raise ValueError(
            f'Invalid model name for ESMC: '
            f'{pretrained_model_name_or_path} '
            f'Valid model names are: {", ".join(embedding_sizes.keys())}.',
            'Or you can set the embedding_size parameter explicitly ',
            'if you are using a fine-tuned model.',
        )

    return embedding_sizes[pretrained_model_name_or_path]


class EsmCambrianEncoderConfig(EncoderConfig):
    """Config for the ESM-Cambrian encoder."""

    # The name of the encoder
    name: Literal['esmc'] = 'esmc'

    pretrained_model_name_or_path: str = Field(
        default='EvolutionaryScale/esmc-300m-2024-12',
        description='The model id, options '
        '[EvolutionaryScale/esmc-300m-2024-12, '
        'EvolutionaryScale/esmc-600m-2024-12]',
    )
    embedding_size: int | None = Field(
        default=None,
        description='The model embedding size. If you are using a '
        'fine-tuned model you should explicitly set this value.',
    )

    @model_validator(mode='after')
    def set_embedding_size(self) -> Self:
        """Set the embedding size based on the model name."""
        self.embedding_size = _validate_embedding_size(
            self.pretrained_model_name_or_path,
            self.embedding_size,
        )

        return self


class EsmCambrianEncoder(Encoder):
    """Encoder for the ESM-Cambrian model.

    For more information on the ESM-Cambrian model, see:
    https://www.evolutionaryscale.ai/blog/esm-cambrian
    """

    def __init__(
        self,
        pretrained_model_name_or_path: str = 'EvolutionaryScale/esmc-600m-2024-12',  # noqa E501
        embedding_size: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the ESM-Cambrian encoder.

        Parameters
        ----------
        pretrained_model_name_or_path : str
            The model id, options ['EvolutionaryScale/esmc-300m-2024-12',
            'EvolutionaryScale/esmc-600m-2024-12'], by default
            'EvolutionaryScale/esmc-600m-2024-12'.
        embedding_size : int, optional
            The model embedding size. If you are using a fine-tuned model you
            should explicitly set this value, by default None.
        **kwargs : Any
            Additional base arguments, see `Encoder`.
        """
        # Initialize the base encoder
        super().__init__(**kwargs)

        from esm.models.esmc import ESMC
        from esm.tokenization import EsmSequenceTokenizer

        # Loads model and auto set device to cuda and dtype to bfloat16
        model = ESMC.from_pretrained(pretrained_model_name_or_path)

        # Set the model to evaluation mode
        model.eval()

        # Load the tokenizer
        tokenizer = EsmSequenceTokenizer()

        # Set the model max length for proper truncation
        tokenizer.model_max_length = 2048

        # Get the embedding size based on the model name
        embedding_size = _validate_embedding_size(
            pretrained_model_name_or_path,
            embedding_size,
        )

        # Set persistent attributes
        self.model = model
        self._tokenizer = tokenizer
        self._embedding_size = embedding_size

    @property
    def dtype(self) -> torch.dtype:
        """Get the data type of the encoder."""
        # NOTE: The model is set to bfloat16 in the ESMC class
        # but we cast to float16 in the encode function to avoid
        # issues with casting in the calling code
        return torch.float16

    @property
    def device(self) -> torch.device:
        """Get the device of the encoder."""
        return self.model.device

    @property
    def embedding_size(self) -> int:
        """Get the embedding size of the encoder."""
        return self._embedding_size

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
        outputs = self.model(sequence_tokens=batch_encoding['input_ids'])

        # Return the last hidden state (cast from bfloat16 to float16)
        return outputs.embeddings.to(torch.float16)
