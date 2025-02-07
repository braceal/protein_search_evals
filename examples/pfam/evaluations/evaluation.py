"""Evaluate a model on the Pfam benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from pydantic import Field

from protein_search_evals.datasets.pfam import Pfam20Dataset
from protein_search_evals.embed.encoders import EncoderConfigs
from protein_search_evals.embed.encoders import Esm2EncoderConfig
from protein_search_evals.embed.encoders import EsmCambrianEncoderConfig
from protein_search_evals.search import FaissIndexConfig
from protein_search_evals.search import Retriever
from protein_search_evals.search import RetrieverConfig
from protein_search_evals.utils import BaseConfig


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
    if 'esm2' in model_name:
        # Initialize encoder configuration
        encoder_config = Esm2EncoderConfig(
            normalize_embeddings=True,
            pretrained_model_name_or_path=model_name,
        )
        return encoder_config
    elif 'esmc' in model_name:
        # Initialize encoder configuration
        encoder_config = EsmCambrianEncoderConfig(
            normalize_embeddings=True,
            pretrained_model_name_or_path=model_name,
        )
        return encoder_config

    else:
        raise ValueError(f'Unknown encoder: {model_name}')


class EvaluationMetadata(BaseConfig):
    """Evaluation metadata dataclass."""

    sequence_level_mean_accuracy: float = Field(
        ...,
        description='The sequence level mean accuracy of the model.',
    )
    family_level_mean_accuracy: float = Field(
        ...,
        description='The family level mean accuracy of the model.',
    )
    sequence_level_median_accuracy: float = Field(
        ...,
        description='The sequence level median accuracy of the model.',
    )
    family_level_median_accuracy: float = Field(
        ...,
        description='The family level median accuracy of the model.',
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
        description='The directory containing the Pfam dataset.',
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
            f'\tFamily-level mean Accuracy: {self.family_level_mean_accuracy * 100:.2f}%\n'  # noqa E501
            f'\tSequence-level median Accuracy: {self.sequence_level_median_accuracy * 100:.2f}%\n'  # noqa E501
            f'\tFamily-level median Accuracy: {self.family_level_median_accuracy * 100:.2f}%\n'  # noqa E501
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
    family_level_mean_accuracy: float = Field(
        ...,
        description='The family level mean accuracy of the model.',
    )
    sequence_level_median_accuracy: float = Field(
        ...,
        description='The sequence level median accuracy of the model.',
    )
    family_level_median_accuracy: float = Field(
        ...,
        description='The family level median accuracy of the model.',
    )
    accuracy_by_seq: dict[str, float] = Field(
        ...,
        description='The accuracy of the model for each sequence.',
    )
    accuracy_by_family: dict[str, float] = Field(
        ...,
        description='The accuracy of the model for each family.',
    )


class Evaluator:
    """Evaluator for the Pfam benchmark."""

    def __init__(
        self,
        dataset: Pfam20Dataset,
        retriever: Retriever,
        top_k: int,
    ) -> None:
        """Initialize the evaluator.

        Parameters
        ----------
        dataset : Pfam20Dataset
            The Pfam dataset.
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
        is the correct family. We exclude self-hits from the
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

        # Get the mapping from uid to family
        uid_to_family = self.dataset.uniprot_to_family

        # Store a mapping from uid to accuracy
        accuracy_by_seq = {}
        for query_sequence, indices in zip(sequences, predicted_indices):
            # Get the query uniprot id and correct family
            query_uid = query_sequence.tag
            correct_family = uid_to_family[query_uid]

            # Get predicted uids
            uids = query_tags[indices]

            # Exclude self-hit from predicted uniprot ids
            predicted_uids = [uid for uid in uids if uid != query_uid]

            # Get the predicted families
            predicted_families = [uid_to_family[uid] for uid in predicted_uids]

            # Get the correct family
            correct_family = uid_to_family[query_sequence.tag]

            # Correct if the top hit is the same pfam
            correct = float(predicted_families[0] == correct_family)
            accuracy_by_seq[query_uid] = correct

        return accuracy_by_seq

    def compute_family_level_accuracy(
        self,
        accuracy_by_seq: dict[str, float],
    ) -> dict[str, float]:
        """Compute the average accuracy within each family.

        Returns
        -------
        dict[str, float]
            The accuracy of the model on the Pfam benchmark
            for each family.
        """
        # Get the mapping from family to list of Uniprot IDs
        families = self.dataset.load_families()

        # Create a dictionary mapping the family name to the accuracy
        accuracy_by_family = {
            family: float(np.mean([accuracy_by_seq[uid] for uid in uids]))
            for family, uids in families.items()
        }

        return accuracy_by_family

    def _compute_avg_accuracy(self, accuracies: dict[str, float]) -> float:
        """Compute the average accuracy.

        Parameters
        ----------
        accuracies : dict[str, float]
            The accuracy of the model for each sequence or family.

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
            The accuracy of the model for each sequence or family.

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

        # Search the index for the nearest neighbors
        # We are only interested in the top hit (excluding self-hit)
        results = self.retriever.search(
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

        # Compute the family level accuracy
        accuracy_by_family = self.compute_family_level_accuracy(
            accuracy_by_seq=accuracy_by_seq,
        )

        # Compute the average sequence/family level mean accuracies
        sequence_level_accuracy = self._compute_avg_accuracy(accuracy_by_seq)
        family_level_accuracy = self._compute_avg_accuracy(accuracy_by_family)

        # Compute the average sequence/family level median accuracies
        sequence_level_median_accuracy = self._compute_median_accuracy(
            accuracy_by_seq,
        )
        family_level_median_accuracy = self._compute_median_accuracy(
            accuracy_by_family,
        )
        # Create the evaluation output
        return EvaluatorOutput(
            sequence_level_mean_accuracy=sequence_level_accuracy,
            family_level_mean_accuracy=family_level_accuracy,
            sequence_level_median_accuracy=sequence_level_median_accuracy,
            family_level_median_accuracy=family_level_median_accuracy,
            accuracy_by_seq=accuracy_by_seq,
            accuracy_by_family=accuracy_by_family,
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
    args = parser.parse_args()

    # Get the dataset directory (a single directory named with a UUID
    # present in the embeddings directory)
    dataset_dir = next((args.model_dir / 'embeddings').glob('*'))

    # Will automatically create the faiss index if it does not exist
    faiss_index_path = args.model_dir / f'{args.precision}-faiss.index'

    # Initialize FaissIndexConfig
    faiss_config = FaissIndexConfig(
        dataset_dir=dataset_dir,
        faiss_index_path=faiss_index_path,
        precision=args.precision,
        search_algorithm='exact',
        search_gpus=0,  # Defaults to GPU 0 as relative to CUDA_VISIBLE_DEVICES
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

    # Load the query sequences
    dataset = Pfam20Dataset(args.dataset_dir)

    # Create the evaluator
    evaluator = Evaluator(dataset, retriever, top_k=2)

    # Run the evaluation
    output = evaluator.run()

    # Create the evaluation metadata
    metadata = EvaluationMetadata(
        sequence_level_mean_accuracy=output.sequence_level_mean_accuracy,
        family_level_mean_accuracy=output.family_level_mean_accuracy,
        sequence_level_median_accuracy=output.sequence_level_median_accuracy,
        family_level_median_accuracy=output.family_level_median_accuracy,
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
