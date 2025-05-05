"""Analyze BLAST results and generate a summary report."""

from __future__ import annotations

from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

import numpy as np
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

    # Then, take the first (top) hit per query (since the subjects
    # are already pre-sorted by pident/bitscore)
    top_hits = non_self_hits.groupby('qseqid', as_index=False).first()

    # Get the ground truth and predicted cluster ids for the queries and hits
    groundtruths = [uid_to_cluster[x] for x in top_hits['qseqid']]
    preds = [uid_to_cluster[x] for x in top_hits['sseqid']]

    # Compute the accuracy of each prediction
    correct = [float(p == c) for p, c in zip(preds, groundtruths)]

    # Compute the accuracy per cluster
    cluster_to_correct = defaultdict(list)
    for cluster, correct_ in zip(groundtruths, correct):
        cluster_to_correct[cluster].append(correct_)

    # Map the cluster id to the accuracy
    accuracies = {
        cluster: np.mean(correct_)
        for cluster, correct_ in cluster_to_correct.items()
    }

    # Compute the mean accuracy statistics
    sequence_level_mean_accuracy = float(np.mean(correct))
    cluster_level_mean_accuracy = float(np.mean(list(accuracies.values())))

    # Compute the median accuracy statistics
    sequence_level_median_accuracy = float(np.median(correct))
    cluster_level_median_accuracy = float(np.median(list(accuracies.values())))

    # Print the results
    print('Sequence level mean accuracy:', sequence_level_mean_accuracy)
    print('Cluster level mean accuracy:', cluster_level_mean_accuracy)
    print('Sequence level median accuracy:', sequence_level_median_accuracy)
    print('Cluster level median accuracy:', cluster_level_median_accuracy)
