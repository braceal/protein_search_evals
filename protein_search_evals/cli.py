"""Command line interface."""

from __future__ import annotations

from pathlib import Path

import typer
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from natsort import natsorted
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
    output_dir.mkdir(parents=True, exist_ok=True)
    record_iter = SeqIO.parse(input_file, 'fasta')
    batch = []
    file_index = 0

    def _write_batch(batch: list[SeqRecord]) -> None:
        # Closure on the file_index
        filename = f'{input_file.stem}_{file_index:04}{input_file.suffix}'
        output_path = output_dir / filename
        SeqIO.write(batch, output_path, 'fasta')

    # Iterate over the records in the fasta file
    # and write them to the output directory in batches
    for record in tqdm(record_iter, desc='Writing sequences'):
        batch.append(record)
        if len(batch) >= num_seqs_per_file:
            _write_batch(batch)
            batch = []
            file_index += 1

    # Write any remaining sequences
    if batch:
        _write_batch(batch)


def main() -> None:
    """Entry point for CLI."""
    app()


if __name__ == '__main__':
    main()
