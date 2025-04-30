"""Command line interface."""

from __future__ import annotations

from pathlib import Path

import typer
from natsort import natsorted

app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)


@app.command()
def merge(
    dataset_dir: Path = typer.Option(  # noqa: B008
        ...,
        '--dataset_dir',
        '-d',
        help='The directory containing the dataset subdirectories '
        'to merge (will glob * this directory).',
    ),
    output_dir: Path = typer.Option(  # noqa: B008
        ...,
        '--output_dir',
        '-o',
        help='The dataset directory to save the merged datasets to.',
    ),
) -> None:
    """Merge datasets from multiple directories output by `generate`."""
    from protein_search_evals.embed.writers import HuggingFaceWriter

    # Initialize the writer
    writer = HuggingFaceWriter()

    # Get the dataset directories
    dataset_dirs = natsorted(dataset_dir.glob('*'))

    # Merge the datasets
    writer.merge(dataset_dirs, output_dir)


@app.command()
def chunk_fasta_file(
    input_file: Path = typer.Option(  # noqa: B008
        ...,
        '--input_file',
        '-i',
        help='The fasta file to chunk.',
    ),
    output_dir: Path = typer.Option(  # noqa: B008
        ...,
        '--output_dir',
        '-o',
        help='The directory to save the chunked fasta files to.',
    ),
    num_seqs_per_file: int = typer.Option(
        ...,
        '--num_seqs_per_file',
        '-n',
        help='The number of sequences per chunked fasta file.',
    ),
) -> None:
    """Chunk a fasta file into smaller fasta files."""
    from protein_search_evals.utils import chunk_fasta_file

    # Chunk the fasta file
    chunk_fasta_file(input_file, output_dir, num_seqs_per_file)


def main() -> None:
    """Entry point for CLI."""
    app()


if __name__ == '__main__':
    main()
