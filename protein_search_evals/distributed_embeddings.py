"""Distributed inference for generating embeddings."""

from __future__ import annotations

import functools
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

from parsl.concurrent import ParslPoolExecutor
from pydantic import Field
from pydantic import field_validator

from protein_search_evals.embed import EncoderConfigs
from protein_search_evals.parsl import ComputeConfigs
from protein_search_evals.utils import BaseConfig


def embedding_worker(
    input_path: Path,
    output_dir: Path,
    encoder_kwargs: dict[str, Any],
    token_embedding_output_dir: Path | None = None,
    token_embedding_buffer_size: int = 50_000,
) -> None:
    """Embed a single file and save a numpy array with embeddings."""
    # Imports are here since this function is called in a parsl process

    from uuid import uuid4

    from protein_search_evals.embed import get_encoder
    from protein_search_evals.embed.embeddings import HDF5TokenEmbeddings
    from protein_search_evals.embed.writers import HuggingFaceWriter
    from protein_search_evals.timer import Timer
    from protein_search_evals.utils import read_fasta

    # Time the worker function
    timer = Timer('finished-embedding', input_path).start()

    # Initialize the model and tokenizer
    with Timer('loaded-encoder', input_path):
        encoder = get_encoder(encoder_kwargs, register=True)

    # Read the sequences from the fasta file
    with Timer('loaded-dataset', input_path):
        fasta_contents = read_fasta(input_path)
        sequences = [x.sequence for x in fasta_contents]

    # Create an output directory name to link the dense
    # embeddings to the pooled embeddings (and metadata)
    dataset_name = str(uuid4())

    # Check if the token embeddings should be saved
    if token_embedding_output_dir is not None:
        token_embedding_writer = HDF5TokenEmbeddings(
            file=token_embedding_output_dir / f'{dataset_name}.hdf5',
            buffer_size=token_embedding_buffer_size,
            max_sequence_length=encoder.max_length,
        )
    else:
        token_embedding_writer = None

    # Compute the embeddings
    with Timer('computed-embeddings', input_path):
        embeddings = encoder.compute_embeddings(
            sequences=sequences,
            token_embedding_writer=token_embedding_writer,
        )

        # Check if the token embeddings should be saved
        if token_embedding_writer is not None:
            token_embedding_writer.flush()

    # Write the result to disk
    with Timer('wrote-embeddings', input_path):
        # Create the output directory for the embedding dataset
        dataset_dir = output_dir / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)

        # Create the result dictionary
        result = {
            'embeddings': embeddings,
            'sequences': sequences,
            'tags': [x.tag for x in fasta_contents],
        }

        # Write the result to disk
        HuggingFaceWriter().write(dataset_dir, result)

    # Stop the timer to log the worker time
    timer.stop()


class Config(BaseConfig):
    """Configuration for distributed inference."""

    input_dir: Path = Field(
        ...,
        description='Directory containing the input files to embed.',
    )
    output_dir: Path = Field(
        ...,
        description='Directory to save the embeddings.',
    )
    glob_patterns: list[str] = Field(
        default=['*'],
        description='Glob patterns to match the input files.',
    )
    store_token_embeddings: bool = Field(
        default=False,
        description='Whether to store the token embeddings.',
    )
    token_embedding_buffer_size: int = Field(
        default=50_000,
        description='The buffer size for writing token embeddings. '
        'The number of embeddings to store in memory before writing to disk.',
    )
    encoder_config: EncoderConfigs = Field(
        ...,
        description='Configuration for the encoder.',
    )
    compute_config: ComputeConfigs = Field(
        ...,
        description='Configuration for the Parsl compute backend.',
    )

    @field_validator('input_dir', 'output_dir')
    @classmethod
    def resolve_path(cls, value: Path) -> Path:
        """Resolve the path to an absolute path."""
        return value.resolve()


if __name__ == '__main__':
    # Parse arguments from the command line
    parser = ArgumentParser(description='Embed sequences.')
    parser.add_argument(
        '--config',
        type=Path,
        required=True,
        help='Path to the .yaml configuration file',
    )
    args = parser.parse_args()

    # Load the configuration
    config = Config.from_yaml(args.config)

    # Create a directory for the embeddings
    embedding_dir = config.output_dir / 'embeddings'

    # Create a directory for the token embeddings if needed
    token_embedding_dir = None
    if config.store_token_embeddings:
        token_embedding_dir = config.output_dir / 'token_embeddings'
        token_embedding_dir.mkdir(parents=True, exist_ok=True)

    # Make the output directory
    embedding_dir.mkdir(parents=True, exist_ok=True)

    # Log the configuration
    config.write_yaml(config.output_dir / 'config.yaml')

    # Set the static arguments of the worker function
    worker_fn = functools.partial(
        embedding_worker,
        output_dir=embedding_dir,
        encoder_kwargs=config.encoder_config.model_dump(),
        token_embedding_output_dir=token_embedding_dir,
        token_embedding_buffer_size=config.token_embedding_buffer_size,
    )

    # Collect all input files
    input_files = []
    for pattern in config.glob_patterns:
        input_files.extend(list(config.input_dir.glob(pattern)))

    # Log the input files to stdout
    print(f'Found {len(input_files)} input files to embed')

    # Set the parsl compute settings
    parsl_config = config.compute_config.get_config(
        config.output_dir / 'parsl',
    )

    # Distribute the input files across processes
    with ParslPoolExecutor(parsl_config) as pool:
        list(pool.map(worker_fn, input_files))
