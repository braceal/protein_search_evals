from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import faiss
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='FAISS IVF binary search speed test',
    )
    parser.add_argument(
        '--index_path',
        type=str,
        default='/dev/shm/test-1B-index.bin',
        help='Path to the FAISS binary index file',
    )
    parser.add_argument(
        '--size',
        type=str,
        default='1B',
        help='Size of the index',
    )
    parser.add_argument(
        '--output_dir',
        type=Path,
        default='results',
        help='Directory to save the JSON results files',
    )
    parser.add_argument(
        '-k',
        type=int,
        default=20,
        help='Number of nearest neighbors to search for',
    )
    parser.add_argument(
        '--dim',
        type=int,
        default=2560,
        help='Embedding dimension (default: 2560 for ESM-2 3B)',
    )
    args = parser.parse_args()

    return args


def run_search_experiment(
    index: faiss.IndexBinaryIVF,
    xq: np.ndarray,
    index_path: str,
    nprobe: int,
    k: int,
    n_queries: int,
    dim: int,
    json_out: str,
) -> None:
    # Set the number of probes for the index
    index.nprobe = nprobe

    search_times = []
    for i in range(5):
        start = time.perf_counter()
        distances, indices = index.search(xq, k)
        elapsed = time.perf_counter() - start
        search_times.append(elapsed)
        print(
            f'Run {i + 1}: CPU binary IVF search: '
            f'{distances.shape}, {indices.shape}, {elapsed:.4f} seconds',
        )

    mean_time = np.mean(search_times)
    std_time = np.std(search_times)
    print(
        'Average search time over 5 runs: '
        f'{mean_time:.4f} ± {std_time:.4f} seconds',
    )

    # Save results to JSON
    results = {
        'index_path': index_path,
        'nprobe': nprobe,
        'k': k,
        'n_queries': n_queries,
        'embedding_dim': dim,
        'mean_search_time': mean_time,
        'std_search_time': std_time,
    }
    with open(json_out, 'w') as f:
        json.dump(results, f, indent=2)


def main() -> None:
    # Parse arguments
    args = parse_args()

    print(f'Running search experiments for index: {args.index_path}')

    # Read the index once
    start = time.perf_counter()
    index = faiss.read_index_binary(args.index_path)
    elapsed = time.perf_counter() - start
    print(f'Read index time: {elapsed:.4f} seconds')

    # Set the values for nprobe
    nprobe_values = [1, 2, 4, 8, 16, 32, 64, 128, 256]

    # Set the values for n_queries
    # (single query and number of proteins in E. coli reference proteome)
    n_queries_values = [1, 4402]

    # Create the output directory if it doesn't exist
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Run the search experiment for each n_queries value
    for n_queries in n_queries_values:
        # Generate random queries
        xq = np.random.randint(
            0,
            256,
            size=(n_queries, args.dim // 8),
            dtype='uint8',
        )
        # Run the search experiment for each nprobe value
        for nprobe in nprobe_values:
            # Set the output file name
            filename = (
                f'results_size_{args.size}_nprobe_{nprobe}_nq_{n_queries}.json'
            )
            run_search_experiment(
                index=index,
                xq=xq,
                index_path=args.index_path,
                nprobe=nprobe,
                k=args.k,
                n_queries=n_queries,
                dim=args.dim,
                json_out=args.output_dir / filename,
            )


if __name__ == '__main__':
    main()
