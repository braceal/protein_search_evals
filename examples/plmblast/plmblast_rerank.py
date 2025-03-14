"""pLM-BLAST example."""

from __future__ import annotations

import argparse
from pathlib import Path

from protein_search_evals.embed import get_encoder
from protein_search_evals.evaluate import get_dataset
from protein_search_evals.evaluate import get_encoder_config
from protein_search_evals.rerankers.plmblast.plmblast import PlmBlastReranker
from protein_search_evals.search import FaissIndex
from protein_search_evals.search import FaissIndexConfig
from protein_search_evals.search import Retriever

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
    search_gpus = 0 if args.gpus == 0 else list(range(1, args.gpus))

    # Initialize faiss index configuration
    faiss_config = FaissIndexConfig(
        dataset_dir=embedding_dataset_dir,
        faiss_index_path=faiss_index_path,
        precision=args.precision,
        search_algorithm='exact',
        search_gpus=search_gpus,
    )

    # Get the encoder configuration
    encoder_config = get_encoder_config(args.model_name)

    # Initialize the encoder
    encoder = get_encoder(encoder_config.model_dump())

    # Initialize the faiss index
    faiss_index = FaissIndex(**faiss_config.model_dump())

    # Initialize the reranker
    reranker = PlmBlastReranker(encoder)

    # Initialize the retriever
    retriever = Retriever(
        encoder=encoder,
        faiss_index=faiss_index,
        reranker=reranker,
    )

    # Load the benchmark dataset containing the  query sequences
    dataset = get_dataset(args.dataset_dir, args.dataset_partition)

    # TODO: finish
