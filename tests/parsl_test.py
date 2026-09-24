"""Tests for Parsl compute configurations."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
from parsl.executors import HighThroughputExecutor
from parsl.launchers import SingleNodeLauncher
from parsl.providers import PBSProProvider
from pydantic import TypeAdapter
from pydantic import ValidationError

from genslm_embeddings.parsl import ComputeConfigs
from genslm_embeddings.parsl import SwingConfig


def test_swing_config_builds_pbs_config(tmp_path: Path) -> None:
    """Configure one single-GPU worker per Swing PBS block."""
    settings = SwingConfig(
        account='project',
        min_blocks=1,
        max_blocks=4,
        worker_init='source .venv/bin/activate',
        address='login.example.org',
    )

    config = settings.get_config(tmp_path)
    executor = cast(HighThroughputExecutor, config.executors[0])
    provider = executor.provider

    assert config.retries == 1
    assert executor.available_accelerators == ['0']
    assert executor.max_workers_per_node == 1
    assert executor.cores_per_worker == 16
    assert executor.address == 'login.example.org'
    assert isinstance(provider, PBSProProvider)
    assert provider.account == 'project'
    assert provider.queue == 'gpu'
    assert provider.nodes_per_block == 1
    assert provider.cpus_per_node == 16
    assert provider.select_options == 'ngpus=1'
    assert provider.worker_init == 'source .venv/bin/activate'
    assert provider.max_blocks == 4
    assert isinstance(provider.launcher, SingleNodeLauncher)


def test_swing_config_parses_from_compute_config_union() -> None:
    """Select Swing through the distributed embedding config union."""
    settings: ComputeConfigs = TypeAdapter(ComputeConfigs).validate_python(
        {'name': 'swing', 'account': 'project', 'queue': 'backfill'},
    )

    assert isinstance(settings, SwingConfig)


@pytest.mark.parametrize(
    ('values', 'message'),
    (
        ({'walltime': '1:00'}, 'walltime must use HH:MM:SS format'),
        (
            {'queue': 'backfill', 'walltime': '04:00:01'},
            'no more than 4:00:00',
        ),
        ({'walltime': '24:00:01'}, 'no more than 24:00:00'),
        ({'max_blocks': 5}, 'less than or equal to 4'),
        ({'min_blocks': 2, 'max_blocks': 1}, 'min_blocks cannot exceed'),
        ({'init_blocks': 2, 'max_blocks': 1}, 'init_blocks cannot exceed'),
    ),
)
def test_swing_config_validates_scheduler_limits(
    values: dict[str, object],
    message: str,
) -> None:
    """Reject settings outside Swing scheduler limits."""
    with pytest.raises(ValidationError, match=message):
        SwingConfig.model_validate({'account': 'project', **values})
