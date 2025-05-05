"""Analyze BLAST results and generate a summary report."""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

from protein_search_evals.evaluate import get_dataset

if __name__ == '__main__':
    # Parse arguments from the command line
    parser = ArgumentParser(description='Analyze BLAST results.')
    parser.add_argument(
        '--blast_log_file',
        type=Path,
        required=True,
        help='Path to the BLAST log file.',
    )
    parser.add_argument(
        '--dataset_dir',
        type=Path,
        required=True,
        help='The directory containing the dataset.',
    )
    parser.add_argument(
        '--dataset_partition',
        type=str,
        default='',
        help='The partition of the dataset to use.',
    )
    args = parser.parse_args()

    # Load the BLAST log file
    text = Path(args.blast_log_file).read_text().splitlines()

    # Get the dataset
    dataset = get_dataset(args.dataset_dir, args.dataset_partition)

    # Get the ground truth clusters
    clusters = dataset.load_clusters()

    # Get the mapping from uid to cluster
    uid_to_cluster = dataset.uniprot_to_cluster

    # Load the BLAST log file into a DataFrame
    df = pd.read_csv(
        filepath_or_buffer=args.blast_log_file,
        sep='\t',
        header=None,
        skiprows=1,  # Skip first line (a nohup message)
        names=[
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

    # We need to skip the self hit (which will be the top hit for each query
    # with qseqid == sseqid and pident == 100) e.g.,
    # qseqid        sseqid        pident    length   mismatch  ...
    # A0A8J7XFM5.1  A0A8J7XFM5.1  100.000   239.0    0.0  ...
    # A0A8J7XFM5.1  A0A8J7XXH7.1  28.750    80.0     56.0  ...

    # First, drop self-hits
    non_self_hits = df[df['qseqid'] != df['sseqid']]

    # Then, take the first (top) hit per query
    top_hits = non_self_hits.groupby('qseqid', as_index=False).first()

    # breakpoint()
