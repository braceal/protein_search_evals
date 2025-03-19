"""Evaluate a model on the Pfam benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from pydantic import Field

from protein_search_evals.datasets.pfam import Pfam20Dataset
from protein_search_evals.datasets.radicalsam import RadicalSamDataset
from protein_search_evals.embed.encoders import EncoderConfigs
from protein_search_evals.embed.encoders import Esm2EncoderConfig
from protein_search_evals.embed.encoders import EsmCambrianEncoderConfig
from protein_search_evals.embed.encoders import ProtTransEncoderConfig
from protein_search_evals.search import FaissIndexConfig
from protein_search_evals.search import Retriever
from protein_search_evals.search import RetrieverConfig
from protein_search_evals.utils import BaseConfig


def get_dataset(
    dataset_dir: str | Path,
    partition: str,
) -> Pfam20Dataset | RadicalSamDataset:
    """Get the Pfam dataset.

    Parameters
    ----------
    dataset_dir : str | Path
        The directory containing the dataset.
    partition : str
        The partition of the dataset to use.

    Returns
    -------
    Pfam20Dataset | RadicalSamDataset
        The dataset.
    """
    if 'pfam' in str(dataset_dir):
        return Pfam20Dataset(dataset_dir)
    elif 'radicalsam' in str(dataset_dir):
        return RadicalSamDataset(dataset_dir, partition=partition)
    else:
        raise ValueError(f'Unknown dataset: {dataset_dir}')


def get_encoder_config(model_name: str) -> EncoderConfigs:
    """Get the encoder configuration.

    Parameters
    ----------
    model_name : str
        The model name to use for the encoder.

    Returns
    -------
    EncoderConfigs
        The encoder configuration.
    """
    # Initialize encoder configuration
    if 'esm2' in model_name:
        return Esm2EncoderConfig(
            normalize_pooled_embeddings=True,
            pretrained_model_name_or_path=model_name,
        )
    elif 'esmc' in model_name:
        return EsmCambrianEncoderConfig(
            normalize_pooled_embeddings=True,
            pretrained_model_name_or_path=model_name,
        )
    elif 'prot_t5' in model_name:
        return ProtTransEncoderConfig(
            normalize_pooled_embeddings=True,
            pretrained_model_name_or_path=model_name,
        )
    else:
        raise ValueError(f'Unknown encoder: {model_name}')


class EvaluationMetadata(BaseConfig):
    """Evaluation metadata dataclass."""

    sequence_level_mean_accuracy: float = Field(
        ...,
        description='The sequence level mean accuracy of the model.',
    )
    cluster_level_mean_accuracy: float = Field(
        ...,
        description='The cluster level mean accuracy of the model.',
    )
    cluster_level_median_accuracy: float = Field(
        ...,
        description='The cluster level median accuracy of the model.',
    )
    precision: str = Field(
        ...,
        description='The precision of the faiss index [float32, ubinary].',
    )
    model: str = Field(
        ...,
        description='The model name to use for the encoder.',
    )
    model_directory: str = Field(
        ...,
        description='The model output directory containing the '
        'embeddings subdir.',
    )
    dataset: str = Field(
        ...,
        description='The directory containing the dataset.',
    )
    index_path: str = Field(
        ...,
        description='The path to the faiss index.',
    )

    def __str__(self) -> str:
        """Return the string representation of the metadata."""
        return (
            f'EvaluationMetadata(\n'
            f'\tSequence-level mean Accuracy: {self.sequence_level_mean_accuracy * 100:.2f}%\n'  # noqa E501
            f'\tCluster-level mean Accuracy: {self.cluster_level_mean_accuracy * 100:.2f}%\n'  # noqa E501
            f'\tCluster-level median Accuracy: {self.cluster_level_median_accuracy * 100:.2f}%\n'  # noqa E501
            f'\tPrecision: {self.precision}\n'
            f'\tModel: {self.model}\n'
            f'\tModel Directory: {self.model_directory}\n'
            f'\tDataset: {self.dataset}\n'
            f'\tIndex Path: {self.index_path}\n'
            f')'
        )


class EvaluatorOutput(BaseConfig):
    """Evaluator output dataclass."""

    sequence_level_mean_accuracy: float = Field(
        ...,
        description='The sequence level mean accuracy of the model.',
    )
    cluster_level_mean_accuracy: float = Field(
        ...,
        description='The cluster level mean accuracy of the model.',
    )
    cluster_level_median_accuracy: float = Field(
        ...,
        description='The cluster level median accuracy of the model.',
    )
    accuracy_by_seq: dict[str, float] = Field(
        ...,
        description='The accuracy of the model for each sequence.',
    )
    accuracy_by_cluster: dict[str, float] = Field(
        ...,
        description='The accuracy of the model for each cluster.',
    )


class Evaluator:
    """Evaluator for the Pfam benchmark."""

    def __init__(
        self,
        dataset: Pfam20Dataset | RadicalSamDataset,
        retriever: Retriever,
        top_k: int,
    ) -> None:
        """Initialize the evaluator.

        Parameters
        ----------
        dataset : Pfam20Dataset | RadicalSamDataset
            The dataset to evaluate the model on.
        retriever : Retriever
            The retriever to use for searching the index.
        top_k : int
            The number of hits to retrieve.
        """
        self.dataset = dataset
        self.retriever = retriever
        self.top_k = top_k

    def _compute_accuracy_by_seq(
        self,
        query_tags: np.ndarray,
        predicted_indices: list[list[int]],
    ) -> dict[str, float]:
        """Compute the accuracy for each sequence (Uniprot ID).

        For each query sequence, we check whether the top hit
        is the correct cluster. We exclude self-hits from the
        predicted uniprot ids.

        Parameters
        ----------
        query_tags : np.ndarray
            The query tags (Uniprot IDs).
        predicted_indices : list[list[int]]
            The predicted indices.

        Returns
        -------
        dict[str, float]
            A mapping from Uniprot ID of the sequence to accuracy (0 or 1).
        """
        # Load the sequences
        sequences = self.dataset.load_sequences()

        # Get the mapping from uid to cluster
        uid_to_cluster = self.dataset.uniprot_to_cluster

        # Store a mapping from uid to accuracy
        accuracy_by_seq = {}
        for query_sequence, indices in zip(sequences, predicted_indices):
            # Get the query uniprot id and correct cluster
            query_uid = query_sequence.tag
            correct_cluster = uid_to_cluster[query_uid]

            # Get predicted uids
            uids = query_tags[indices]

            # Exclude self-hit from predicted uniprot ids
            predicted_uids = [uid for uid in uids if uid != query_uid]

            # Get the predicted clusters
            predicted_clusters = [
                uid_to_cluster[uid] for uid in predicted_uids
            ]

            # Get the correct cluster
            correct_cluster = uid_to_cluster[query_sequence.tag]

            # Correct if the top hit is the same pfam
            correct = float(predicted_clusters[0] == correct_cluster)
            accuracy_by_seq[query_uid] = correct

        return accuracy_by_seq

    def compute_cluster_level_accuracy(
        self,
        accuracy_by_seq: dict[str, float],
    ) -> dict[str, float]:
        """Compute the average accuracy within each cluster.

        Returns
        -------
        dict[str, float]
            The accuracy of the model on the benchmark for each cluster.
        """
        # Get the mapping from cluster to list of Uniprot IDs
        clusters = self.dataset.load_clusters()

        # Create a dictionary mapping the cluster name to the accuracy
        accuracy_by_cluster = {
            cluster: float(np.mean([accuracy_by_seq[uid] for uid in uids]))
            for cluster, uids in clusters.items()
        }

        return accuracy_by_cluster

    def _compute_avg_accuracy(self, accuracies: dict[str, float]) -> float:
        """Compute the average accuracy.

        Parameters
        ----------
        accuracies : dict[str, float]
            The accuracy of the model for each sequence or cluster.

        Returns
        -------
        float
            The average accuracy.
        """
        return float(np.mean(list(accuracies.values())))

    def _compute_median_accuracy(self, accuracies: dict[str, float]) -> float:
        """Compute the median accuracy.

        Parameters
        ----------
        accuracies : dict[str, float]
            The accuracy of the model for each sequence or cluster.

        Returns
        -------
        float
            The median accuracy.
        """
        return float(np.median(list(accuracies.values())))

    def run(self) -> EvaluatorOutput:
        """Run the evaluation on the Pfam dataset.

        Returns
        -------
        EvaluatorOutput
            The evaluation output.
        """
        # Number of sequences in the full dataset
        num_sequences = len(self.retriever.faiss_index.dataset)

        # The evaluation is over all sequences
        query_keys = np.arange(num_sequences)

        # Get all the sequence tags (Uniprot IDs)
        query_tags = self.retriever.get(query_keys, key='tags')

        # Get all query embeddings from the index
        query_embeddings = self.retriever.get(query_keys, key='embeddings')

        # Get all the sequences
        sequences = self.retriever.get(query_keys, key='sequences').tolist()

        # Search the index for the nearest neighbors
        # We are only interested in the top hit (excluding self-hit)
        results = self.retriever.search(
            query=sequences,
            query_embedding=query_embeddings,
            top_k=self.top_k,
        )

        # Get the predicted sequence indices (labels)
        predicted_indices = results[0].total_indices

        # Compute the accuracy by Uniprot ID
        accuracy_by_seq = self._compute_accuracy_by_seq(
            query_tags=query_tags,
            predicted_indices=predicted_indices,
        )

        # Compute the cluster level accuracy
        accuracy_by_cluster = self.compute_cluster_level_accuracy(
            accuracy_by_seq=accuracy_by_seq,
        )

        # Compute the average sequence/cluster level mean accuracies
        sequence_level_acc = self._compute_avg_accuracy(accuracy_by_seq)
        cluster_level_acc = self._compute_avg_accuracy(accuracy_by_cluster)

        # Compute the average cluster level median accuracies
        cluster_level_median_acc = self._compute_median_accuracy(
            accuracy_by_cluster,
        )
        # Create the evaluation output
        return EvaluatorOutput(
            sequence_level_mean_accuracy=sequence_level_acc,
            cluster_level_mean_accuracy=cluster_level_acc,
            cluster_level_median_accuracy=cluster_level_median_acc,
            accuracy_by_seq=accuracy_by_seq,
            accuracy_by_cluster=accuracy_by_cluster,
        )


if __name__ == '__main__':
    # Parse the arguments
    parser = argparse.ArgumentParser(
        description='Evaluate a model on the Pfam benchmark.',
    )
    parser.add_argument(
        '--report_name',
        type=str,
        required=True,
        help='The name prefix of the report files to save.',
    )
    parser.add_argument(
        '--dataset_dir',
        type=Path,
        required=True,
        help='The directory containing the Pfam dataset.',
    )
    parser.add_argument(
        '--dataset_partition',
        type=str,
        default='',
        help='The partition of the dataset to use.',
    )
    parser.add_argument(
        '--model_dir',
        type=Path,
        required=True,
        help='The model output directory containing the embeddings subdir.',
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default='facebook/esm2_t6_8M_UR50D',
        help='The model name to use for the encoder.',
    )
    parser.add_argument(
        '--precision',
        type=str,
        default='float32',
        help='The precision of the faiss index [float32, ubinary].',
    )
    parser.add_argument(
        '--gpus',
        type=int,
        default=0,
        help='The number of GPUs to use for searching the faiss index.',
    )
    args = parser.parse_args()

    # Get the dataset directory (a single directory named with a UUID
    # present in the embeddings directory)
    embedding_dataset_dir = next((args.model_dir / 'embeddings').glob('*'))

    # Will automatically create the faiss index if it does not exist
    faiss_index_path = args.model_dir / f'{args.precision}-faiss.index'

    # The encoder model always gets placed on GPU:0 relative
    # to CUDA_VISIBLE_DEVICES. If `gpus` > 0, then the faiss index
    # will be placed on the next available GPUs (relative to
    # CUDA_VISIBLE_DEVICES). Otherwise, the faiss index will share
    # the same GPU as the encoder.
    search_gpus = 0 if args.gpus == 0 else list(range(1, args.gpus + 1))

    # Initialize FaissIndexConfig
    faiss_config = FaissIndexConfig(
        dataset_dir=embedding_dataset_dir,
        faiss_index_path=faiss_index_path,
        precision=args.precision,
        search_algorithm='exact',
        search_gpus=search_gpus,
    )

    # Get the encoder configuration
    encoder_config = get_encoder_config(args.model_name)

    # Initialize retriever configuration
    retriever_config = RetrieverConfig(
        faiss_config=faiss_config,
        encoder_config=encoder_config,
    )

    # Create the retriever
    retriever = retriever_config.get_retriever()

    # Load the benchmark dataset containing the  query sequences
    dataset = get_dataset(args.dataset_dir, args.dataset_partition)

    # Create the evaluator
    evaluator = Evaluator(dataset, retriever, top_k=2)

    # Run the evaluation
    output = evaluator.run()

    # Create the evaluation metadata
    metadata = EvaluationMetadata(
        sequence_level_mean_accuracy=output.sequence_level_mean_accuracy,
        cluster_level_mean_accuracy=output.cluster_level_mean_accuracy,
        cluster_level_median_accuracy=output.cluster_level_median_accuracy,
        precision=args.precision,
        model=args.model_name,
        model_directory=str(args.model_dir),
        dataset=str(args.dataset_dir),
        index_path=str(faiss_index_path),
    )

    # Print the evaluation summary
    print('Evaluation Summary:')
    print(metadata)

    # Save the metadata to a file
    metadata.write_json(f'{args.report_name}_metadata.json')

    # Save the output to a file
    output.write_json(f'{args.report_name}_output.json')
