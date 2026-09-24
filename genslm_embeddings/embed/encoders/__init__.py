"""Encoder module."""

from __future__ import annotations

from typing import Any
from typing import Union

from parsl_object_registry import registry

from genslm_embeddings.embed.encoders.base import Encoder
from genslm_embeddings.embed.encoders.esm2 import Esm2Encoder
from genslm_embeddings.embed.encoders.esm2 import Esm2EncoderConfig
from genslm_embeddings.embed.encoders.esmc import EsmCambrianEncoder
from genslm_embeddings.embed.encoders.esmc import EsmCambrianEncoderConfig
from genslm_embeddings.embed.encoders.prottrans import ProtTransEncoder
from genslm_embeddings.embed.encoders.prottrans import ProtTransEncoderConfig

EncoderConfigs = Union[
    Esm2EncoderConfig,
    EsmCambrianEncoderConfig,
    ProtTransEncoderConfig,
]

STRATEGIES = {
    'esm2': Esm2Encoder,
    'esmc': EsmCambrianEncoder,
    'prottrans': ProtTransEncoder,
}


# This is a workaround to support optional registration.
# Make a function to combine the config and instance initialization
# since the registry only accepts functions with hashable arguments.
def _factory_fn(**kwargs: dict[str, Any]) -> Encoder:
    name = kwargs.pop('name', '')
    cls = STRATEGIES.get(name)  # type: ignore[arg-type]
    if not cls:
        raise ValueError(
            f'Unknown encoder name: {name}.'
            f' Available: {set(STRATEGIES.keys())}',
        )

    return cls(**kwargs)


def get_encoder(
    kwargs: dict[str, Any],
    register: bool = False,
) -> Encoder:
    """Get the instance based on the kwargs.

    Currently supports the following strategies:
    - esm2
    - esmc
    - prottrans

    Parameters
    ----------
    kwargs : dict[str, Any]
        The configuration. Contains a `name` argument
        to specify the strategy to use.
    register : bool, optional
        Register the instance for warmstart. Caches the
        instance based on the kwargs, by default False.

    Returns
    -------
    Encoder
        The instance.

    Raises
    ------
    ValueError
        If the `name` is unknown.
    """
    # Create and register the instance
    if register:
        registry.register(_factory_fn)
        registry_kwargs = dict(kwargs)
        pooled_layers = registry_kwargs.get('pooled_layers')
        if isinstance(pooled_layers, list):
            registry_kwargs['pooled_layers'] = tuple(pooled_layers)
        return registry.get(_factory_fn, **registry_kwargs)

    return _factory_fn(**kwargs)
