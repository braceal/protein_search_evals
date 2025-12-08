"""Analyze the disorder of a Pfam family using AIUPred."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from tqdm import tqdm

from protein_search_evals.datasets.pfam import Pfam20Dataset

# Add the AIUPred library to the path (it should be in the same directory as
# this file)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import aiupred_lib


def parse_args() -> argparse.Namespace:
    """Parse arguments from the command line."""
    parser = argparse.ArgumentParser(
        description='Analyze the disorder of a Pfam family using AIUPred.',
    )
    parser.add_argument(
        '--dataset_dir',
        type=Path,
        required=True,
        help='The directory containing the dataset.',
    )
    return parser.parse_args()


if __name__ == '__main__':
    # Parse arguments from the command line
    args = parse_args()

    # Load the Pfam20 dataset
    dataset = Pfam20Dataset(args.dataset_dir)
    sequences = dataset.load_sequences()

    # Load the models and let AIUPred find if a GPU is available.
    embedding_model, regression_model, device = aiupred_lib.init_models(
        'disorder',
    )

    # Collect the predictions for each sequence
    predictions = []

    # Predict disorder of a sequence
    for sequence in tqdm(sequences, desc='Predicting disorder'):
        # Run AIUPred to predict the disorder of the sequence
        prediction = aiupred_lib.predict_disorder(
            sequence.sequence,
            embedding_model,
            regression_model,
            device,
            smoothing=True,
        )

        # Add the prediction to the predictions list
        predictions.append(prediction.tolist())

    # Write the predictions to a json file with the sequence tag as the key
    # and the prediction as the value
    results = [
        {
            'tag': sequence.tag,
            'sequence': sequence.sequence,
            'disorder_prediction': prediction,
        }
        for sequence, prediction in zip(sequences, predictions)
    ]

    # Write the results to a json file
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
