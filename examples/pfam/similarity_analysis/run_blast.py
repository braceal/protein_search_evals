"""Run BLAST."""

from __future__ import annotations

import functools
import shutil
import subprocess
import tempfile
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def run_blast(fasta_file: Path, output_dir: Path) -> None:
    """Run BLAST."""
    # Create a temporary directory on the shared memory for all operations
    with tempfile.TemporaryDirectory(dir='/dev/shm') as temp_dir:
        temp_dir_path = Path(temp_dir)

        # Set the temporary output directory structure
        temp_run_dir = temp_dir_path / fasta_file.stem
        temp_target_db_dir = temp_run_dir / 'target_db'
        temp_target_db_dir.mkdir(parents=True)

        # Run the makeblastdb command
        command = (
            f'makeblastdb -in {fasta_file} -dbtype prot '
            f'-out {temp_target_db_dir}'
        )
        result = subprocess.run(
            command.split(),
            check=False,
            capture_output=True,
        )

        # Log the makeblastdb output
        makeblastdb_log_file = temp_run_dir / 'makeblastdb.log'
        makeblastdb_log_file.write_text(result.stdout.decode('utf-8'))

        # Run the blastp command
        command = (
            f'blastp -query {fasta_file} -db {temp_target_db_dir} '
            '-outfmt 6 -evalue 1e6 -max_target_seqs 1000 -max_hsps 1 '
            '-num_threads 1'
        )
        result = subprocess.run(
            command.split(),
            check=False,
            capture_output=True,
        )

        # Parse the blastp output
        blastp_output = result.stdout.decode('utf-8').splitlines()
        blastp_data = [line.split('\t') for line in blastp_output]
        blastp_df = pd.DataFrame(
            blastp_data,
            columns=[
                'qseqid',
                'sseqid',
                'pident',
                'length',
                'mismatch',
                'gapopen',
                'qstart',
                'qend',
                'sstart',
                'send',
                'evalue',
                'bitscore',
            ],
        )
        blastp_df.to_csv(
            temp_run_dir / f'{fasta_file.stem}_blastp.csv',
            index=False,
        )

        # Move the entire temporary run directory to the persistent output dir
        final_run_dir = output_dir / fasta_file.stem
        shutil.move(str(temp_run_dir), str(final_run_dir))


def main() -> None:
    """Run BLAST."""
    # Parse arguments from the command line
    parser = ArgumentParser(
        description='Run BLAST.',
    )
    parser.add_argument(
        '--input_dir',
        type=Path,
        required=True,
        help='The directory containing the FASTA files to run BLAST on.',
    )
    parser.add_argument(
        '--output_dir',
        type=Path,
        required=True,
        help='The directory to write the BLAST results to.',
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        required=True,
        help='The number of workers to use for the BLAST search.',
    )
    args = parser.parse_args()

    # Create the output directory if it doesn't exist
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Fasta input files
    fasta_files = list(args.input_dir.glob('*.fasta'))

    # Setup a partial worker function
    worker_fn = functools.partial(run_blast, output_dir=args.output_dir)

    # Run BLAST for each FASTA file in parallel
    with ProcessPoolExecutor(max_workers=args.num_workers) as pool:
        for _ in tqdm(
            pool.map(worker_fn, fasta_files),
            desc='Running BLAST',
        ):
            pass


if __name__ == '__main__':
    main()
