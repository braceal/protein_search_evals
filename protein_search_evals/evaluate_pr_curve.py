"""Compute precision-recall curves for homology detection on the benchmark.

Offline plotting example (after saving JSON with this script):

    import json
    import matplotlib.pyplot as plt

    with open('report_esm2_float32_seed-42_pr_curve.json') as f:
        data = json.load(f)
    plt.figure(figsize=(8, 6))
    plt.plot(data['recall'], data['precision'], marker='.',
             label=f"{data['method']} (AUPRC = {data['auprc']:.2f})")
    plt.title('Precision-Recall Curve for Homology Detection')
    plt.xlabel('Recall (Sensitivity)')
    plt.ylabel('Precision (PPV)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from pydantic import Field
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve

from protein_search_evals.datasets.pfam import Pfam20Dataset
from protein_search_evals.datasets.radicalsam import RadicalSamDataset
from protein_search_evals.evaluate import get_dataset
from protein_search_evals.evaluate import get_encoder_config
from protein_search_evals.search import FaissIndexConfig
from protein_search_evals.search import Retriever
from protein_search_evals.search import RetrieverConfig
from protein_search_evals.utils import BaseConfig


class PRCurveOutput(BaseConfig):
    """Output for PR curve evaluation (saved for offline plotting)."""

    precision: list[float] = Field(
        ...,
        description='Precision at each threshold (length = len(recall)).',
    )
    recall: list[float] = Field(
        ...,
        description='Recall at each threshold (length = len(precision)).',
    )
    thresholds: list[float] = Field(
        ...,
        description='Decreasing score thresholds '
        '(length = len(precision) - 1).',
    )
    auprc: float = Field(
        ...,
        description='Area under the precision-recall curve '
        '(average precision).',
    )
    method: str = Field(
        ...,
        description='Model/method name for legend when plotting.',
    )
    precision_type: str = Field(
        ...,
        description='FAISS index precision (e.g. float32, ubinary).',
    )
    dataset: str = Field(
        ...,
        description='Dataset directory used for evaluation.',
    )
    n_pairs: int = Field(
        ...,
        description='Number of (query, retrieved) pairs used '
        '(excluding self-hits).',
    )
    n_positive: int = Field(
        ...,
        description='Number of homologous pairs (same cluster) in the '
        'evaluated set.',
    )


def run_pr_curve_evaluation(
    dataset: Pfam20Dataset | RadicalSamDataset,
    retriever: Retriever,
    top_k: int,
    method_name: str,
    precision_type: str,
    dataset_dir: str | Path,
) -> PRCurveOutput:
    """Build (y_true, y_scores) from retrieval results and compute PR curve.

    For each query sequence, we consider all top_k retrieved hits (excluding
    self). Each (query, retrieved) pair is labeled 1 (homolog) if they are
    in the same cluster, else 0. Scores are the retriever similarity scores
    (higher = more similar). For E-value-based methods you would use
    y_scores = -np.log10(e_values + 1e-300) so that smaller E-values become
    larger scores before calling precision_recall_curve.

    Parameters
    ----------
    dataset : Pfam20Dataset | RadicalSamDataset
        Benchmark dataset with uniprot_to_cluster.
    retriever : Retriever
        Retriever used for search (returns .total_scores, .total_indices).
    top_k : int
        Number of neighbors retrieved per query (more => more pairs for PR).
    method_name : str
        Label for the method (e.g. model name).
    precision_type : str
        FAISS precision (e.g. float32, ubinary).
    dataset_dir : str | Path
        Path to dataset directory (for metadata).

    Returns
    -------
    PRCurveOutput
        Precision, recall, thresholds, AUPRC, and metadata for offline
        plotting.
    """
    num_sequences = len(retriever.faiss_index.dataset)
    query_keys = np.arange(num_sequences)
    query_tags = retriever.get(query_keys, key='tags')
    query_embeddings = retriever.get(query_keys, key='embeddings')
    sequences = retriever.get(query_keys, key='sequences').tolist()

    results, _ = retriever.search(
        query=sequences,
        query_embedding=query_embeddings,
        top_k=top_k,
    )

    uid_to_cluster = dataset.uniprot_to_cluster

    y_true_list, y_scores_list = [], []

    for i, (indices, scores) in enumerate(
        zip(results.total_indices, results.total_scores),
    ):
        query_uid = query_tags[i]
        query_cluster = uid_to_cluster[query_uid]

        for retrieved_idx, score in zip(indices, scores):
            if retrieved_idx == i:
                continue
            retrieved_uid = query_tags[retrieved_idx]
            retrieved_cluster = uid_to_cluster[retrieved_uid]
            label = 1 if query_cluster == retrieved_cluster else 0
            y_true_list.append(label)
            y_scores_list.append(float(score))

    y_true = np.array(y_true_list, dtype=np.int64)
    y_scores = np.array(y_scores_list, dtype=np.float64)

    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    auprc = average_precision_score(y_true, y_scores)

    return PRCurveOutput(
        precision=precision.tolist(),
        recall=recall.tolist(),
        thresholds=thresholds.tolist(),
        auprc=float(auprc),
        method=method_name,
        precision_type=precision_type,
        dataset=str(dataset_dir),
        n_pairs=len(y_true),
        n_positive=int(y_true.sum()),
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compute precision-recall curve data for homology '
        'detection.',
    )
    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Output path for PR curve JSON '
        '(e.g. report_esm2_float32_pr_curve.json).',
    )
    parser.add_argument(
        '--dataset_dir',
        type=Path,
        required=True,
        help='Directory containing the dataset.',
    )
    parser.add_argument(
        '--dataset_partition',
        type=str,
        default='',
        help='Partition of the dataset to use.',
    )
    parser.add_argument(
        '--model_dir',
        type=Path,
        required=True,
        help='Model output directory containing the embeddings subdir.',
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default='facebook/esm2_t6_8M_UR50D',
        help='Model name for the encoder.',
    )
    parser.add_argument(
        '--precision',
        type=str,
        default='float32',
        help='FAISS index precision [float32, ubinary].',
    )
    parser.add_argument(
        '--top_k',
        type=int,
        default=500,
        help='Number of neighbors per query for building PR pairs '
        '(default: 500).',
    )
    parser.add_argument(
        '--gpus',
        type=int,
        default=0,
        help='Number of GPUs for FAISS search.',
    )
    args = parser.parse_args()

    embedding_dataset_dir = next((args.model_dir / 'embeddings').glob('*'))
    faiss_index_path = args.model_dir / f'{args.precision}-faiss.index'
    search_gpus = 0 if args.gpus == 0 else list(range(1, args.gpus + 1))

    faiss_config = FaissIndexConfig(
        dataset_dir=embedding_dataset_dir,
        faiss_index_path=faiss_index_path,
        precision=args.precision,
        search_algorithm='exact',
        search_gpus=search_gpus,
    )
    encoder_config = get_encoder_config(args.model_name)
    retriever_config = RetrieverConfig(
        faiss_config=faiss_config,
        encoder_config=encoder_config,
    )
    retriever = retriever_config.get_retriever()
    dataset = get_dataset(args.dataset_dir, args.dataset_partition)

    output = run_pr_curve_evaluation(
        dataset=dataset,
        retriever=retriever,
        top_k=args.top_k,
        method_name=args.model_name,
        precision_type=args.precision,
        dataset_dir=args.dataset_dir,
    )

    print(f'AUPRC: {output.auprc:.4f}')
    print(f'Pairs: {output.n_pairs}, Positives: {output.n_positive}')

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.write_json(args.output)
    print(f'Wrote PR curve data to {args.output}')
