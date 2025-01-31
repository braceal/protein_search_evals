"""Utilities to build Parsl configurations."""

from __future__ import annotations

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
from parsl.providers import LocalProvider
from parsl.providers import PBSProProvider
from pydantic import BaseModel
from pydantic import Field


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


ComputeConfigs = Union[WorkstationConfig, PolarisConfig]
