"""Build a FAISS exact binary index for benchmarking."""

from __future__ import annotations

import argparse
import time

import faiss
import numpy as np
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Build a FAISS exact binary index for benchmarking',
    )
    parser.add_argument(
        '--npoints',
        type=int,
        default=250_000_000,
        help='Number of data points to generate and index',
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output filename for the FAISS index (e.g., '
        'speed-test-1B-exact-index.bin)',
    )
    parser.add_argument(
        '--dim',
        type=int,
        default=2560,
        help='Embedding dimension (default: 2560 for ESM-2 3B)',
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1_000_000,
        help='Batch size for adding vectors to index',
    )
    return parser.parse_args()


def main() -> None:
    """Build a FAISS exact binary index for benchmarking."""
    args = parse_args()

    # --- Prepare binary data -----------------------------------------
    xb = np.random.randint(
        0,
        256,
        size=(args.npoints, args.dim // 8),
        dtype='uint8',
    )

    # --- Build the index --------------------------------------------
    # Exact binary index (Hamming distance)
    index = faiss.IndexBinaryFlat(args.dim)

    print(
        f'Adding {args.npoints:,} vectors to index in batches '
        f'of {args.batch_size:,}',
    )
    start = time.perf_counter()

    # Add vectors in batches
    num_batches = (args.npoints + args.batch_size - 1) // args.batch_size
    for i in tqdm(range(num_batches)):
        batch_start = i * args.batch_size
        batch_end = min((i + 1) * args.batch_size, args.npoints)
        batch = xb[batch_start:batch_end]
        index.add(batch)

    elapsed = time.perf_counter() - start
    print(f'Total add time: {elapsed:.2f}s')

    print(f'Writing index to {args.output}')
    faiss.write_index_binary(index, args.output)


if __name__ == '__main__':
    main()
