"""Command line interface."""

from __future__ import annotations

from pathlib import Path

import typer
from tqdm import tqdm

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
    dataset_dirs = list(dataset_dir.glob('*'))

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
    num_chunks: int = typer.Option(
        ...,
        '--chunk_size',
        '-c',
        help='The number of smaller files to chunk the fasta file into.',
    ),
) -> None:
    """Chunk a fasta file into smaller fasta files."""
    from protein_search_evals.utils import batch_data
    from protein_search_evals.utils import read_fasta
    from protein_search_evals.utils import write_fasta

    # Read the fasta file
    sequences = read_fasta(input_file)

    # Chunk the sequences
    chunks = batch_data(sequences, len(sequences) // num_chunks)

    # Make the output directory
    output_dir.mkdir(parents=True)

    # Save the chunked fasta files
    for i, chunk in tqdm(enumerate(chunks), desc='Writing chunks'):
        filename = f'{input_file.stem}_{i:04}{input_file.suffix}'
        write_fasta(chunk, output_dir / filename)


def main() -> None:
    """Entry point for CLI."""
    app()


if __name__ == '__main__':
    main()
