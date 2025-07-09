"""Build a FAISS binary IVF index for benchmarking."""

from __future__ import annotations

import argparse
import time

import faiss
import numpy as np


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Build a FAISS binary IVF index for benchmarking',
    )
    parser.add_argument(
        '--npoints',
        type=int,
        default=250_000_000,
        help='Number of data points to generate and index',
    )
    parser.add_argument(
        '--nlist',
        type=int,
        default=2048,
        help='Number of IVF lists (clusters) for the index',
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output filename for the FAISS index (e.g., '
        'speed-test-1B-index.bin)',
    )
    parser.add_argument(
        '--dim',
        type=int,
        default=2560,
        help='Embedding dimension (default: 2560 for ESM-2 3B)',
    )
    parser.add_argument(
        '--ivf_max_train_size',
        type=int,
        default=2_000_000,
        help='Maximum number of embeddings to use for training',
    )
    return parser.parse_args()


def main() -> None:
    """Build a FAISS binary IVF index for benchmarking."""
    args = parse_args()

    # --- Prepare binary data -----------------------------------------
    xb = np.random.randint(
        0,
        256,
        size=(args.npoints, args.dim // 8),
        dtype='uint8',
    )

    # --- Build the index --------------------------------------------
    # Coarse quantizer (Hamming)
    quant = faiss.IndexBinaryFlat(args.dim)
    # Binary IVF index
    cpu_idx = faiss.IndexBinaryIVF(quant, args.dim, args.nlist)
    # Set the number of probes for the index (doesn't matter for saving)
    cpu_idx.nprobe = 4

    # Limit the number of embeddings used for training
    embeddings = xb
    train_size = min(len(embeddings), args.ivf_max_train_size)
    if train_size < len(embeddings):
        print(
            f'Using {train_size:,} embeddings for training '
            f'(randomly sampled from {len(embeddings):,})',
        )
        # Randomly sample embeddings for training
        indices = np.random.choice(
            len(embeddings),
            size=train_size,
            replace=False,
        )
        train_embeddings = embeddings[indices]
    else:
        train_embeddings = embeddings

    print('Training')
    start = time.perf_counter()
    cpu_idx.train(train_embeddings)  # train centroids in Hamming space
    elapsed = time.perf_counter() - start
    print(f'train time: {elapsed}')

    start = time.perf_counter()
    cpu_idx.add(xb)  # add vectors to lists
    elapsed = time.perf_counter() - start
    print(f'add time: {elapsed}')

    faiss.write_index_binary(cpu_idx, args.output)
