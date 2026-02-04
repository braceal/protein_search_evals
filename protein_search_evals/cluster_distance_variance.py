"""Compute per-cluster variance of pairwise distances for Pfam embeddings.

Loads each model's embeddings and, for each Pfam cluster, computes the
variance of the pairwise distance matrix (cosine similarity or Hamming
distance) to measure how spread out the cluster is in embedding space.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from datasets import load_from_disk
from pydantic import Field
from scipy.spatial.distance import pdist
from tqdm import tqdm

from protein_search_evals.datasets.pfam import Pfam20Dataset
from protein_search_evals.evaluate import get_dataset
from protein_search_evals.utils import BaseConfig


class ClusterVarianceResult(BaseConfig):
    """Per-cluster variance and summary for one distance metric."""

    per_cluster: dict[str, dict[str, float | int]] = Field(
        ...,
        description='Map cluster_id -> {variance, n_sequences}.',
    )
    mean_variance: float = Field(
        ...,
        description='Mean variance across clusters.',
    )
    median_variance: float = Field(
        ...,
        description='Median variance across clusters.',
    )


class ClusterVarianceOutput(BaseConfig):
    """Full output: both metrics and metadata."""

    model: str = Field(..., description='Model name.')
    dataset: str = Field(..., description='Dataset directory.')
    embedding_dir: str = Field(..., description='Embedding dataset path used.')
    n_sequences: int = Field(
        ...,
        description='Total sequences in embedding set.',
    )
    n_clusters: int = Field(
        ...,
        description='Number of clusters with ≥2 sequences.',
    )
    cosine_similarity: ClusterVarianceResult = Field(
        ...,
        description='Variance of pairwise cosine similarities per cluster.',
    )
    hamming_distance: ClusterVarianceResult = Field(
        ...,
        description='Variance of pairwise Hamming distances '
        '(binary embeddings).',
    )


def _variance_cosine_similarity(embeddings: np.ndarray) -> float:
    """Variance of pairwise cosine similarities (upper triangle, no diagonal).

    Assumes embeddings are L2-normalized so dot product = cosine similarity.
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized_embeddings = embeddings / norms
    sim = normalized_embeddings @ normalized_embeddings.T
    triu = np.triu_indices(sim.shape[0], k=1)
    values = sim[triu]
    return float(np.var(values))


def _variance_hamming_distance(embeddings: np.ndarray) -> float:
    """Variance of pairwise Hamming distances (fraction of differing bits).

    Binarizes embeddings by sign (positive -> 1, else 0), then computes
    pairwise Hamming fraction per pair.
    """
    binary = (embeddings > 0).astype(np.uint8)
    # pdist(..., 'hamming') = fraction of coordinates that differ
    d = pdist(binary, metric='hamming')
    return float(np.var(d))


def compute_cluster_variances(
    embedding_dir: Path,
    dataset_dir: Path,
    dataset_partition: str,
    model_name: str,
) -> ClusterVarianceOutput:
    """Compute variance per cluster for both metrics.

    Parameters
    ----------
    embedding_dir : Path
        Path to the HuggingFace embedding dataset
            (e.g. model_dir/embeddings/<uuid>).
    dataset_dir : Path
        Pfam dataset directory (e.g. data/pfam/).
    dataset_partition : str
        Dataset partition (e.g. '' for Pfam, or 'seed-42' style for subset).
    model_name : str
        Model name for the output metadata.

    Returns
    -------
    ClusterVarianceOutput
        Per-cluster variances for cosine similarity and Hamming distance.
    """
    dataset_pfam = get_dataset(str(dataset_dir), dataset_partition)
    if not isinstance(dataset_pfam, Pfam20Dataset):
        raise ValueError(
            'Cluster variance is implemented for Pfam20Dataset only; '
            f'got {type(dataset_pfam).__name__}',
        )

    ds = load_from_disk(str(embedding_dir))
    ds.set_format('numpy')

    uid_to_index = {ds['tags'][i]: i for i in range(len(ds))}

    clusters = dataset_pfam.load_clusters()
    cluster_to_indices: dict[str, list[int]] = {}
    for cid, uids in clusters.items():
        indices = [uid_to_index[uid] for uid in uids if uid in uid_to_index]
        if len(indices) >= 2:
            cluster_to_indices[cid] = indices

    embeddings_full = np.array(ds['embeddings'][:], dtype=np.float64)
    n_sequences = len(embeddings_full)

    cosine_per_cluster: dict[str, dict[str, float | int]] = {}
    hamming_per_cluster: dict[str, dict[str, float | int]] = {}

    for cid, indices in tqdm(
        cluster_to_indices.items(),
        desc='Cluster variance',
        unit='cluster',
    ):
        embs = embeddings_full[indices]
        n = len(embs)
        var_cos = _variance_cosine_similarity(embs)
        var_ham = _variance_hamming_distance(embs)
        cosine_per_cluster[cid] = {'variance': var_cos, 'n_sequences': n}
        hamming_per_cluster[cid] = {'variance': var_ham, 'n_sequences': n}

    def _summary(
        per_cluster: dict[str, dict[str, float | int]],
    ) -> ClusterVarianceResult:
        vars_list = [v['variance'] for v in per_cluster.values()]
        return ClusterVarianceResult(
            per_cluster=per_cluster,
            mean_variance=float(np.mean(vars_list)),
            median_variance=float(np.median(vars_list)),
        )

    return ClusterVarianceOutput(
        model=model_name,
        dataset=str(dataset_dir),
        embedding_dir=str(embedding_dir),
        n_sequences=n_sequences,
        n_clusters=len(cluster_to_indices),
        cosine_similarity=_summary(cosine_per_cluster),
        hamming_distance=_summary(hamming_per_cluster),
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compute per-cluster distance variance for Pfam '
        'embeddings.',
    )
    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Output JSON path (e.g. cluster_variance_esm2_8M.json).',
    )
    parser.add_argument(
        '--dataset_dir',
        type=Path,
        required=True,
        help='Pfam dataset directory (e.g. data/pfam/).',
    )
    parser.add_argument(
        '--dataset_partition',
        type=str,
        default='',
        help='Dataset partition (e.g. seed-42).',
    )
    parser.add_argument(
        '--model_dir',
        type=Path,
        required=True,
        help='Model output directory containing embeddings subdir.',
    )
    parser.add_argument(
        '--model_name',
        type=str,
        required=True,
        help='Model name (for output metadata).',
    )
    args = parser.parse_args()

    embedding_dir = next((args.model_dir / 'embeddings').glob('*'))

    result = compute_cluster_variances(
        embedding_dir=embedding_dir,
        dataset_dir=args.dataset_dir,
        dataset_partition=args.dataset_partition,
        model_name=args.model_name,
    )

    print(f'Model: {result.model}')
    print(f'Clusters: {result.n_clusters}, Sequences: {result.n_sequences}')
    print(
        'Cosine similarity - '
        f'mean variance: {result.cosine_similarity.mean_variance:.6f}, '
        f'median: {result.cosine_similarity.median_variance:.6f}',
    )
    print(
        f'Hamming distance - '
        f'mean variance: {result.hamming_distance.mean_variance:.6f}, '
        f'median: {result.hamming_distance.median_variance:.6f}',
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.write_json(args.output)
    print(f'Wrote {args.output}')
