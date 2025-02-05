"""Evaluate a model on the Pfam benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from protein_search_evals.datasets.pfam import Pfam20Dataset
from protein_search_evals.embed.encoders import EncoderConfigs
from protein_search_evals.embed.encoders import Esm2EncoderConfig
from protein_search_evals.embed.encoders import EsmCambrianEncoderConfig
from protein_search_evals.search import FaissIndexConfig
from protein_search_evals.search import Retriever
from protein_search_evals.search import RetrieverConfig
from protein_search_evals.utils import Sequence


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


def compute_sequence_level_accuracy(
    sequences: list[Sequence],
    predicted_indices: list[list[int]],
    query_tags: np.ndarray,
    uid_to_family: dict[str, str],
) -> float:
    """Compute the sequence level accuracy.

    Parameters
    ----------
    sequences : list[Sequence]
        The list of query sequences.
    predicted_indices : list[list[int]]
        The list of predicted indices.
    query_tags : np.ndarray
        The query tags (correct string labels).
    uid_to_family : dict[str, str]
        The mapping from uid to family.

    Returns
    -------
    float
        The accuracy of the model on the Pfam benchmark
        at the sequence level.
    """
    # Check whether the top hit is the correct family
    correctness = []
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
        correct = predicted_families[0] == correct_family
        correctness.append(correct)

    # Compute the accuracy
    accuracy = float(np.mean(correctness))

    return accuracy


def run_evaluation(retriever: Retriever, dataset: Pfam20Dataset) -> float:
    """Run the evaluation on the Pfam dataset.

    Parameters
    ----------
    retriever : Retriever
        The retriever to use for searching the index.

    dataset : Pfam20Dataset
        The dataset to evaluate on.

    Returns
    -------
    float
        The accuracy of the model on the Pfam benchmark.
    """
    # Load the sequences
    sequences = dataset.load_sequences()

    # Get the mapping from uid to family
    uid_to_family = dataset.uniprot_to_family

    # The evaluation is over all sequences
    query_keys = np.arange(len(sequences))

    # Get all the sequence tags (Uniprot IDs)
    query_tags = retriever.get(query_keys, key='tags')

    # Get all query embeddings from the index
    query_embeddings = retriever.get(query_keys, key='embeddings')

    # Search the index for the nearest neighbors
    # We are only interested in the top hit (excluding self-hit)
    results = retriever.search(query_embedding=query_embeddings, top_k=2)

    # Get the predicted sequence indices (labels)
    predicted_indices = results[0].total_indices

    # Check whether the top hit is the correct family
    sequence_level_accuracy = compute_sequence_level_accuracy(
        sequences=sequences,
        predicted_indices=predicted_indices,
        query_tags=query_tags,
        uid_to_family=uid_to_family,
    )

    return sequence_level_accuracy


if __name__ == '__main__':
    # Parse the arguments
    parser = argparse.ArgumentParser(
        description='Evaluate a model on the Pfam benchmark.',
    )
    parser.add_argument(
        '--report_file',
        type=Path,
        required=True,
        help='The file to save the evaluation report to.',
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

    # Run the evaluation
    accuracy = run_evaluation(retriever, dataset)

    # Create an accuracy report
    report = {
        'Accuracy': f'{accuracy * 100:.2f}%',
        'Precision': args.precision,
        'Model': args.model_name,
        'Model Directory': str(args.model_dir),
        'Dataset': str(args.dataset_dir),
        'Index Path': str(faiss_index_path),
    }

    # Print the evaluation summary
    print('Evaluation Summary:')
    for key, value in report.items():
        print(f'\t{key}: {value}')

    # Save the report to a file
    with open(args.report_file, 'w') as f:
        json.dump(report, f)
