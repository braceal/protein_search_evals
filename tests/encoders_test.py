"""Tests for encoder layer pooling."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
import torch
from pydantic import ValidationError
from torch.utils.data import DataLoader
from transformers import BatchEncoding
from transformers import PreTrainedTokenizer

from genslm_embeddings.embed.encoders.base import Encoder
from genslm_embeddings.embed.encoders.base import EncoderConfig
from genslm_embeddings.embed.encoders.base import select_pooled_embeddings
from genslm_embeddings.embed.encoders.esm2 import Esm2Encoder
from genslm_embeddings.embed.encoders.prottrans import ProtTransEncoder


class FakeEncoder(Encoder):
    """Small deterministic encoder that exposes three transformer layers."""

    @property
    def dtype(self) -> torch.dtype:
        """Use float32 for predictable assertions."""
        return torch.float32

    @property
    def device(self) -> torch.device:
        """Run the fake encoder on CPU."""
        return torch.device('cpu')

    @property
    def max_length(self) -> int:
        """Return the fixed test sequence length."""
        return 4

    @property
    def embedding_size(self) -> int:
        """Return the fixed test hidden dimension."""
        return 2

    @property
    def tokenizer(self) -> PreTrainedTokenizer:
        """The overridden dataloader does not use a tokenizer."""
        raise AssertionError('tokenizer should not be accessed')

    @property
    def supports_layer_pooling(self) -> bool:
        """Expose intermediate states for the tests."""
        return True

    @property
    def num_layers(self) -> int:
        """Expose three transformer blocks."""
        return 3

    def get_dataloader(self, sequences: list[str]) -> DataLoader:
        """Return one padded batch for two test sequences."""
        assert sequences == ['AA', 'A']
        batch = BatchEncoding(
            {
                'input_ids': torch.zeros((2, 4), dtype=torch.long),
                'attention_mask': torch.tensor(
                    [[1, 1, 1, 1], [1, 1, 1, 0]],
                ),
            },
        )
        return cast(DataLoader, [batch])

    def _hidden_state(self, layer: int) -> torch.Tensor:
        value = float(layer + 1)
        state = torch.tensor([value, value * 2])
        return state.expand(2, 4, 2).clone()

    def encode(self, batch_encoding: BatchEncoding) -> torch.Tensor:
        """Return the final transformer state."""
        return self._hidden_state(self.num_layers - 1)

    def encode_layers(
        self,
        batch_encoding: BatchEncoding,
        layer_indices: tuple[int, ...],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """Return the final and requested transformer states."""
        return self.encode(batch_encoding), tuple(
            self._hidden_state(layer) for layer in layer_indices
        )


class UnsupportedFakeEncoder(FakeEncoder):
    """Fake an adapter that exposes only its final hidden state."""

    @property
    def supports_layer_pooling(self) -> bool:
        """Do not expose intermediate transformer layers."""
        return False


def test_encoder_defaults_to_final_layer() -> None:
    """Keep the existing two-dimensional final-layer output by default."""
    output = FakeEncoder().compute_embeddings(['AA', 'A'])

    assert output.pool_embeddings.shape == (2, 2)
    assert output.layer_pool_embeddings is None
    assert output.layer_indices is None
    np.testing.assert_allclose(output.pool_embeddings, [[3, 6], [3, 6]])


def test_encoder_pools_all_layers() -> None:
    """Stack every pooled transformer block along the layer axis."""
    output = FakeEncoder(pooled_layers='all').compute_embeddings(['AA', 'A'])

    assert output.layer_pool_embeddings is not None
    assert output.layer_pool_embeddings.shape == (2, 3, 2)
    assert output.layer_indices == (0, 1, 2)
    np.testing.assert_allclose(
        output.layer_pool_embeddings[0],
        [[1, 2], [2, 4], [3, 6]],
    )
    np.testing.assert_allclose(output.pool_embeddings, [[3, 6], [3, 6]])


def test_encoder_preserves_explicit_layer_order_and_normalizes() -> None:
    """Use requested block order and normalize each layer independently."""
    output = FakeEncoder(
        pooled_layers=[2, 0],
        normalize_pooled_embeddings=True,
    ).compute_embeddings(['AA', 'A'])

    assert output.layer_pool_embeddings is not None
    assert output.layer_indices == (2, 0)
    np.testing.assert_allclose(
        np.linalg.norm(output.layer_pool_embeddings, axis=-1),
        np.ones((2, 2)),
    )
    np.testing.assert_allclose(
        output.pool_embeddings,
        output.layer_pool_embeddings[:, 0, :],
    )


def test_encoder_keeps_token_embeddings_final_layer_only() -> None:
    """Do not multiply ragged token output by the selected layer count."""
    output = FakeEncoder(pooled_layers='all').compute_embeddings(
        ['AA', 'A'],
        return_token_embeddings=True,
    )

    assert output.token_embeddings is not None
    assert [embedding.shape for embedding in output.token_embeddings] == [
        (2, 2),
        (1, 2),
    ]
    np.testing.assert_allclose(output.token_embeddings[0], [[3, 6], [3, 6]])


def test_select_pooled_embeddings_uses_requested_transformer_block() -> None:
    """Select query embeddings using transformer block metadata."""
    output = FakeEncoder(pooled_layers=[2, 0]).compute_embeddings(['AA', 'A'])

    selected = select_pooled_embeddings(output, 0)

    assert output.layer_pool_embeddings is not None
    np.testing.assert_array_equal(
        selected,
        output.layer_pool_embeddings[:, 1, :],
    )
    with pytest.raises(ValueError, match='did not produce embedding layer 1'):
        select_pooled_embeddings(output, 1)


@pytest.mark.parametrize('pooled_layers', ([], [-1], [0, 0]))
def test_encoder_config_rejects_invalid_layer_lists(
    pooled_layers: list[int],
) -> None:
    """Reject empty, negative, and duplicate explicit layer selections."""
    with pytest.raises(ValidationError):
        EncoderConfig(pooled_layers=pooled_layers)


def test_encoder_rejects_out_of_range_layer() -> None:
    """Validate explicit indices against the loaded model depth."""
    with pytest.raises(ValueError, match='out of range'):
        FakeEncoder(pooled_layers=[3]).compute_embeddings(['AA', 'A'])


def test_encoder_rejects_unsupported_intermediate_layers() -> None:
    """Fail clearly for adapters without a stable hidden-state API."""
    with pytest.raises(ValueError, match='does not support'):
        UnsupportedFakeEncoder(pooled_layers='all').compute_embeddings(
            ['AA', 'A'],
        )


def test_encoder_rejects_multilayer_token_cache(tmp_path: Path) -> None:
    """A final-layer token cache cannot reconstruct intermediate layers."""
    encoder = FakeEncoder(
        pooled_layers='all',
        cached_token_embeddings_path=tmp_path / 'tokens.h5',
    )

    with pytest.raises(ValueError, match='model inference is required'):
        encoder.compute_embeddings(['AA', 'A'])


class FakeModel:
    """Return a configurable tuple of hidden states."""

    def __init__(self, hidden_states: tuple[torch.Tensor, ...]) -> None:
        self.hidden_states = hidden_states

    def __call__(self, **kwargs: object) -> SimpleNamespace:
        """Return model outputs and verify hidden states were requested."""
        assert kwargs['output_hidden_states'] is True
        return SimpleNamespace(
            hidden_states=self.hidden_states,
            last_hidden_state=self.hidden_states[-1],
        )


def test_esm2_layer_indices_skip_input_embedding_state() -> None:
    """Map block zero to hidden-state tuple position one for ESM-2."""
    states = tuple(torch.full((1, 1, 1), value) for value in range(4))
    encoder = object.__new__(Esm2Encoder)
    encoder.enable_faesm = False
    encoder.model = FakeModel(states)

    final, selected = encoder.encode_layers(BatchEncoding(), (0, 2))

    assert final is states[3]
    assert selected == (states[1], states[3])


def test_prottrans_layer_indices_skip_input_embedding_state() -> None:
    """Map block zero to hidden-state tuple position one for ProtTrans."""
    states = tuple(torch.full((1, 1, 1), value) for value in range(4))
    encoder = object.__new__(ProtTransEncoder)
    encoder.model = FakeModel(states)

    final, selected = encoder.encode_layers(BatchEncoding(), (1, 0))

    assert final is states[3]
    assert selected == (states[2], states[1])
