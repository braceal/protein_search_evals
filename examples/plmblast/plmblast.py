"""pLM-BLAST example."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from protein_search_evals.embed import get_encoder
from protein_search_evals.evaluate import get_encoder_config
from protein_search_evals.rerankers.plmblast.plmblast import PlmBlastReranker
from protein_search_evals.utils import read_fasta

if __name__ == '__main__':
    # Parse the arguments
    parser = argparse.ArgumentParser(description='pLM-BLAST')
    parser.add_argument(
        '--fasta_file',
        type=Path,
        required=True,
        help='The fasta file containing the query sequences.',
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default='facebook/esm2_t6_8M_UR50D',
        help='The model name to use for the encoder.',
    )
    parser.add_argument(
        '--cached_token_embeddings_path',
        type=Path,
        default=None,
        help='The path to the cached HDF5 token embeddings.',
    )
    args = parser.parse_args()

    # Get the encoder configuration
    encoder_config = get_encoder_config(
        args.model_name,
        cached_token_embeddings_path=args.cached_token_embeddings_path,
    )

    # Initialize the encoder
    encoder = get_encoder(encoder_config.model_dump())

    # Initialize the reranker
    reranker = PlmBlastReranker(encoder)

    # Load the sequences to align
    sequences = read_fasta(args.fasta_file)

    # Use the first sequence as the query
    query = sequences[0].sequence
    hits = np.array([x.sequence for x in sequences[1:]])

    # Run the pLM-BLAST search
    results = reranker.rerank(query, hits)

    # Print the results
    for ind in results:
        print(f'{ind} {hits[ind]}\n\n')
