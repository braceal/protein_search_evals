"""Utilities to build Parsl configurations."""

from __future__ import annotations

import os
import re
from abc import ABC
from abc import abstractmethod
from pathlib import Path

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal  # type: ignore [assignment]

from typing import Sequence
from typing import Union

from parsl.addresses import address_by_hostname
from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.launchers import MpiExecLauncher
from parsl.launchers import SingleNodeLauncher
from parsl.launchers import WrappedLauncher
from parsl.providers import LocalProvider
from parsl.providers import PBSProProvider
from pydantic import BaseModel
from pydantic import Field
from pydantic import model_validator

_SWING_CPUS_PER_GPU = 16
_SWING_MAX_RUNNING_JOBS = 4


def _validate_swing_walltime(queue: str, walltime: str) -> None:
    """Validate a walltime against Swing queue limits."""
    match = re.fullmatch(r'(\d+):([0-5]\d):([0-5]\d)', walltime)
    if match is None:
        raise ValueError('walltime must use HH:MM:SS format')

    hours, minutes, seconds = (int(value) for value in match.groups())
    duration_seconds = hours * 3600 + minutes * 60 + seconds
    max_hours = 4 if queue == 'backfill' else 24
    if duration_seconds <= 0 or duration_seconds > max_hours * 3600:
        raise ValueError(
            f'walltime for the {queue} queue must be greater than zero '
            f'and no more than {max_hours}:00:00',
        )


class BaseComputeConfig(BaseModel, ABC):
    """Compute configuration (HPC platform, number of GPUs, etc)."""

    @abstractmethod
    def get_config(self, run_dir: str | Path) -> Config:
        """Create a new Parsl configuration.

        Parameters
        ----------
        run_dir : str | Path
            Path to store monitoring DB and parsl logs.

        Returns
        -------
        Config
            Parsl configuration.
        """
        ...


class WorkstationConfig(BaseComputeConfig):
    """Configuration for a workstation with GPUs."""

    # Name of the configuration
    name: Literal['workstation'] = 'workstation'

    available_accelerators: Union[int, Sequence[str]] = Field(  # noqa UP007
        default=1,
        title='Number of GPUs',
        description='Number of GPU accelerators to use, behaves like '
        'CUDA_VISIBLE_DEVICES.',
    )
    worker_port_range: tuple[int, int] = Field(
        default=(10000, 20000),
        title='Port range',
    )
    retries: int = Field(
        default=1,
        description='Number of retries upon failure.',
    )
    label: str = Field(
        default='htex',
        description='Label for the executor.',
    )

    def get_config(self, run_dir: str | Path) -> Config:
        """Create a parsl configuration for running on a workstation."""
        return Config(
            run_dir=str(run_dir),
            retries=self.retries,
            executors=[
                HighThroughputExecutor(
                    address=address_by_hostname(),
                    label=self.label,
                    cpu_affinity='block',
                    available_accelerators=self.available_accelerators,
                    worker_port_range=self.worker_port_range,
                    provider=LocalProvider(init_blocks=1, max_blocks=1),
                ),
            ],
        )


class SwingConfig(BaseComputeConfig):
    """Configuration for the Swing GPU cluster at Argonne LCRC."""

    name: Literal['swing'] = 'swing'
    account: str = Field(
        min_length=1,
        description='LCRC project to charge for the allocation.',
    )
    queue: Literal['gpu', 'gpu-large', 'backfill'] = Field(
        default='gpu',
        description='Swing PBS queue.',
    )
    walltime: str = Field(
        default='01:00:00',
        description='Requested PBS walltime in HH:MM:SS format.',
    )
    init_blocks: int = Field(
        default=1,
        ge=0,
        description='Number of allocation blocks requested at startup.',
    )
    min_blocks: int = Field(
        default=0,
        ge=0,
        description='Minimum number of active allocation blocks.',
    )
    max_blocks: int = Field(
        default=4,
        ge=1,
        le=_SWING_MAX_RUNNING_JOBS,
        description='Maximum number of active allocation blocks.',
    )
    parallelism: float = Field(
        default=1.0,
        gt=0,
        le=1,
        description='Ratio of provisioned worker slots to active tasks.',
    )
    worker_init: str = Field(
        default='',
        description='Shell commands that prepare the worker environment.',
    )
    scheduler_options: str = Field(
        default='',
        description='Additional PBS directives for the submission script.',
    )
    address: str | None = Field(
        default=None,
        description='Login-node address workers use for the interchange.',
    )
    worker_port_range: tuple[int, int] = Field(
        default=(55000, 56000),
        description='Port range used by workers.',
    )
    interchange_port_range: tuple[int, int] = Field(
        default=(56001, 57000),
        description='Port range used by the Parsl interchange.',
    )
    retries: int = Field(
        default=1,
        ge=0,
        description='Number of retries for failed tasks.',
    )
    heartbeat_threshold: int = Field(
        default=300,
        gt=0,
        description='Seconds to wait before considering a worker lost.',
    )
    label: str = Field(
        default='swing-htex',
        description='Label for the executor.',
    )

    @model_validator(mode='after')
    def validate_scheduler_limits(self) -> SwingConfig:
        """Validate block sizing and queue-specific walltime limits."""
        _validate_swing_walltime(self.queue, self.walltime)

        if self.min_blocks > self.max_blocks:
            raise ValueError('min_blocks cannot exceed max_blocks')
        if self.init_blocks > self.max_blocks:
            raise ValueError('init_blocks cannot exceed max_blocks')

        return self

    def get_config(self, run_dir: str | Path) -> Config:
        """Create a Parsl configuration that submits jobs to Swing."""
        return Config(
            run_dir=str(run_dir),
            retries=self.retries,
            executors=[
                HighThroughputExecutor(
                    label=self.label,
                    address=self.address or address_by_hostname(),
                    cpu_affinity='block',
                    cores_per_worker=_SWING_CPUS_PER_GPU,
                    max_workers_per_node=1,
                    available_accelerators=1,
                    worker_port_range=self.worker_port_range,
                    interchange_port_range=self.interchange_port_range,
                    heartbeat_threshold=self.heartbeat_threshold,
                    provider=PBSProProvider(
                        account=self.account,
                        queue=self.queue,
                        walltime=self.walltime,
                        nodes_per_block=1,
                        cpus_per_node=_SWING_CPUS_PER_GPU,
                        init_blocks=self.init_blocks,
                        min_blocks=self.min_blocks,
                        max_blocks=self.max_blocks,
                        parallelism=self.parallelism,
                        worker_init=self.worker_init,
                        scheduler_options=self.scheduler_options,
                        select_options='ngpus=1',
                        launcher=SingleNodeLauncher(),
                    ),
                ),
            ],
        )


class PolarisConfig(BaseComputeConfig):
    """Polaris@ALCF configuration.

    See here for details: https://docs.alcf.anl.gov/polaris/workflows/parsl/
    """

    name: Literal['polaris'] = 'polaris'

    num_nodes: int = Field(
        default=1,
        description='Number of nodes to request.',
    )
    worker_init: str = Field(
        default='',
        description='How to start a worker. Should load any modules '
        'and environments.',
    )
    scheduler_options: str = Field(
        default='#PBS -l filesystems=home:eagle:grand',
        description='PBS directives, pass -J for array jobs.',
    )
    account: str = Field(
        default='',
        description='The account to charge compute to.',
    )
    queue: str = Field(
        default='',
        description='Which queue to submit jobs to, will usually be prod.',
    )
    walltime: str = Field(
        default='',
        description='Maximum job time, e.g. "01:00:00".',
    )
    cpus_per_node: int = Field(
        default=32,
        description='Number of CPUs per node.',
    )
    cores_per_worker: float = Field(
        default=8,
        description='Number of cores per worker. Evenly distributed '
        'between GPUs.',
    )
    retries: int = Field(
        default=0,
        description='Number of retries upon failure.',
    )
    label: str = Field(
        default='htex',
        description='Label for the executor.',
    )
    worker_debug: bool = Field(
        default=False,
        description='Enable worker debug.',
    )

    def get_config(self, run_dir: str | Path) -> Config:
        """Create a parsl configuration for running on Polaris@ALCF.

        We will launch 4 workers per node, each pinned to a different GPU.

        Parameters
        ----------
        run_dir: str | Path
            Directory in which to store Parsl run files.
        """
        return Config(
            executors=[
                HighThroughputExecutor(
                    label=self.label,
                    heartbeat_period=15,
                    heartbeat_threshold=120,
                    worker_debug=self.worker_debug,
                    # available_accelerators will override settings
                    # for max_workers
                    available_accelerators=4,
                    cores_per_worker=self.cores_per_worker,
                    # address=address_by_interface('bond0'),
                    cpu_affinity='block-reverse',
                    prefetch_capacity=0,
                    provider=PBSProProvider(
                        launcher=MpiExecLauncher(
                            bind_cmd='--cpu-bind',
                            overrides='--depth=64 --ppn 1',
                        ),
                        account=self.account,
                        queue=self.queue,
                        select_options='ngpus=4',
                        scheduler_options=self.scheduler_options,
                        worker_init=self.worker_init,
                        nodes_per_block=self.num_nodes,
                        init_blocks=1,
                        min_blocks=0,
                        max_blocks=1,  # Increase to have more parallel jobs
                        cpus_per_node=self.cpus_per_node,
                        walltime=self.walltime,
                    ),
                ),
            ],
            run_dir=str(run_dir),
            retries=self.retries,
            app_cache=True,
        )


class PolarisHeadlessConfig(BaseComputeConfig):
    """Polaris@ALCF headless configuration.

    See here for details: https://docs.alcf.anl.gov/polaris/workflows/parsl/
    """

    name: Literal['polaris_headless'] = 'polaris_headless'

    num_nodes: int = Field(
        ge=1,
        description='Number of nodes to use (must use at least 1 nodes).',
    )
    retries: int = Field(
        default=1,
        description='Number of retries for the task.',
    )
    max_idletime: float = Field(
        default=60.0 * 10,
        description='The maximum idle time allowed for an executor before '
        'strategy could shut down unused blocks. Default is 10 minutes.',
    )

    def get_config(self, run_dir: str | Path) -> Config:
        """Create a parsl configuration for running headless on Polaris@ALCF.

        We will launch 4 workers per node, each pinned to a different GPU.

        Parameters
        ----------
        run_dir: str | Path
            Directory in which to store Parsl run files.
        """
        # Convert run_dir to a Path object and create the directory
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)

        # Parse the hosts from the nodefile
        with open(os.environ['PBS_NODEFILE']) as fp:
            hosts = [x.strip() for x in fp]

        # Write the hostfile
        hostfile = run_dir / 'htex.hosts'
        hostfile.write_text('\n'.join(hosts))

        # Return the Parsl configuration
        return Config(
            run_dir=str(run_dir),
            retries=self.retries,
            max_idletime=self.max_idletime,
            executors=[
                HighThroughputExecutor(
                    label='htex',
                    cpu_affinity='block-reverse',
                    available_accelerators=4,
                    provider=LocalProvider(
                        launcher=WrappedLauncher(
                            f'mpiexec -n {self.num_nodes} --ppn 1 --hostfile '
                            f'{hostfile} --depth=64 --cpu-bind depth',
                        ),
                        cmd_timeout=120,
                        nodes_per_block=self.num_nodes,
                        init_blocks=1,
                        max_blocks=1,
                    ),
                ),
            ],
        )


ComputeConfigs = Union[
    WorkstationConfig,
    SwingConfig,
    PolarisConfig,
    PolarisHeadlessConfig,
]
