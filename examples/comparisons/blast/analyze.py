"""Analyze BLAST results and generate a summary report."""

from __future__ import annotations

from argparse import ArgumentParser
from collections import defaultdict
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

from protein_search_evals.evaluate import EvaluatorOutput
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
        '--output_file',
        type=Path,
        required=True,
        help='The JSON file to write the output to.',
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

    # Get all unique query sequences from the dataset
    all_query_sequences = set(uid_to_cluster.keys())

    # Pre-parse and filter out invalid (non-tabular) lines
    valid_lines = []
    with open(args.blast_log_file) as f:
        next(f)  # Skip the first line (a nohup message)
        for line in f:
            if len(line.strip().split('\t')) == 12:
                valid_lines.append(line)

    # Load the BLAST log file into a DataFrame
    df = pd.read_csv(
        StringIO(''.join(valid_lines)),
        sep='\t',
        header=None,
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

    # Get sequences that have no hits
    sequences_with_hits = set(top_hits['qseqid'])
    sequences_without_hits = all_query_sequences - sequences_with_hits

    # Get the ground truth and predicted cluster ids for the queries and hits
    groundtruths = [uid_to_cluster[x] for x in top_hits['qseqid']]
    preds = [uid_to_cluster[x] for x in top_hits['sseqid']]

    # Add sequences with no hits as incorrect predictions
    for seq in sequences_without_hits:
        groundtruths.append(uid_to_cluster[seq])
        preds.append('NO_HIT')  # Special marker for sequences with no hits

    # Compute the accuracy of each prediction
    correct = [
        float(p == c) if p != 'NO_HIT' else 0.0
        for p, c in zip(preds, groundtruths)
    ]

    # Compute the accuracy per cluster
    cluster_to_correct = defaultdict(list)
    for cluster, correct_ in zip(groundtruths, correct):
        cluster_to_correct[cluster].append(correct_)

    # Map the cluster id to the accuracy
    accuracy_by_cluster = {
        cluster: float(np.mean(correct_))
        for cluster, correct_ in cluster_to_correct.items()
    }

    # Compute the accuracy by sequence
    accuracy_by_seq = dict(zip(top_hits['qseqid'], correct))

    # Compute the mean accuracy statistics
    sequence_level_mean_accuracy = float(np.mean(correct))
    cluster_level_mean_accuracy = float(
        np.mean(list(accuracy_by_cluster.values())),
    )

    # Compute the median accuracy statistics
    sequence_level_median_accuracy = float(np.median(correct))
    cluster_level_median_accuracy = float(
        np.median(list(accuracy_by_cluster.values())),
    )

    # Print the results
    print('Sequence level mean accuracy:', sequence_level_mean_accuracy)
    print('Cluster level mean accuracy:', cluster_level_mean_accuracy)
    print('Sequence level median accuracy:', sequence_level_median_accuracy)
    print('Cluster level median accuracy:', cluster_level_median_accuracy)

    # Create the evaluation output
    output = EvaluatorOutput(
        sequence_level_mean_accuracy=sequence_level_mean_accuracy,
        cluster_level_mean_accuracy=cluster_level_mean_accuracy,
        cluster_level_median_accuracy=cluster_level_median_accuracy,
        accuracy_by_seq=accuracy_by_seq,
        accuracy_by_cluster=accuracy_by_cluster,
    )

    # Save the output to a file
    output.write_json(args.output_file)
