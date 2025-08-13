# IVF Search Speed Analysis

This directory contains scripts for benchmarking FAISS IVF search performance.


## Index Sizes
- 1M:   `speed-test-1M-index.bin`
- 10M:  `speed-test-10M-index.bin`
- 100M: `speed-test-100M-index.bin`
- 1B:   `speed-test-1B-index.bin`

## Benchmarking Search Time vs. nprobe

This section provides commands to benchmark search time as a function of `nprobe` for two different values of `n_queries` (1 and 4402), with `k=20` for all runs, and for multiple index sizes. Each run saves its results to a separate JSON file for easy plotting.

### Building Indices

First, build the required indices using the `build_index.py` script:

```sh
mkdir -p faiss_indices
python build_index.py --npoints 1000000 --ivf_max_train_size 100000 --nlist 1024 --output faiss_indices/speed-test-1M-index.bin
python build_index.py --npoints 10000000 --ivf_max_train_size 200000 --nlist 4096 --output faiss_indices/speed-test-10M-index.bin
python build_index.py --npoints 100000000 --ivf_max_train_size 1000000 --nlist 16384 --output faiss_indices/speed-test-100M-index.bin
python build_index.py --npoints 1000000000 --ivf_max_train_size 2000000 --nlist 32768 --output faiss_indices/speed-test-1B-index.bin
```

**Note:** The `nlist` parameter is set to the closest power of 2 to the square root of the number of points.

**Note:** The `ivf_max_train_size` parameter is set relatively small to make running the script faster.

**Note:** The indices are saved in the `faiss_indices/` directory, but are not included in this repository,
since they are too large.

### Run Benchmarks

The speed test script automatically runs experiments for all nprobe values (1, 2, 4, 8, 16, 32, 64, 128, 256) and both n_queries values (1, 4402):

```sh
python speed_test.py --index_path faiss_indices/speed-test-1M-index.bin --size 1M --output_dir results
python speed_test.py --index_path faiss_indices/speed-test-10M-index.bin --size 10M --output_dir results
python speed_test.py --index_path faiss_indices/speed-test-100M-index.bin --size 100M --output_dir results
python speed_test.py --index_path faiss_indices/speed-test-1B-index.bin --size 1B --output_dir results
```

Results will be saved to the `results/` directory with filenames like `results_size_1M_nprobe_16_nq_4402.json`.

### Building Exact Index

To build the exact index, run the following command:
```sh
mkdir -p faiss_indices
python build_index_exact.py --npoints 1000000 --output faiss_indices/speed-test-1M-exact-index.bin
python build_index_exact.py --npoints 10000000 --output faiss_indices/speed-test-10M-exact-index.bin
python build_index_exact.py --npoints 100000000 --batch_size 5000000 --output faiss_indices/speed-test-100M-exact-index.bin
python build_index_exact.py --npoints 1000000000 --batch_size 50000000 --output faiss_indices/speed-test-1B-exact-index.bin
```

**Note:** The exact index is saved in the `faiss_indices/` directory, but is not included in this repository, since it is too large.

### Run Benchmarks

```sh
python speed_test_exact.py --index_path faiss_indices/speed-test-1M-exact-index.bin --size 1M --output_dir results_exact
python speed_test_exact.py --index_path faiss_indices/speed-test-10M-exact-index.bin --size 10M --output_dir results_exact
python speed_test_exact.py --index_path faiss_indices/speed-test-100M-exact-index.bin --size 100M --output_dir results_exact
python speed_test_exact.py --index_path faiss_indices/speed-test-1B-exact-index.bin --size 1B --output_dir results_exact
```

Results will be saved to the `results_exact/` directory with filenames like `results_exact_size_1M_k_1_nq_4402.json`.

## Plotting Results

See `analysis.ipynb` for an example of how to plot the results.
