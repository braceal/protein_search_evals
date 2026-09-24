# GenSLM Embeddings

[![CI](https://github.com/ramanathanlab/genslm-embeddings/actions/workflows/ci.yml/badge.svg)](https://github.com/ramanathanlab/genslm-embeddings/actions/workflows/ci.yml)
[![Docs](https://github.com/ramanathanlab/genslm-embeddings/actions/workflows/docs.yml/badge.svg?branch=main)](https://ramanathanlab.github.io/genslm-embeddings/)
[![Release](https://img.shields.io/github/v/release/ramanathanlab/genslm-embeddings?include_prereleases&sort=semver)](https://github.com/ramanathanlab/genslm-embeddings/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

Generate and evaluate biological foundation model embeddings at scale, with a
focus on GenSLM.

📖 **Documentation:** https://ramanathanlab.github.io/genslm-embeddings/

## Installation

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), clone the
repository, and synchronize the locked environment:

```bash
git clone git@github.com:ramanathanlab/genslm-embeddings.git
cd genslm-embeddings
uv sync --locked
```

To install Faiss, for GPU support with CUDA 12, run the following command:
```bash
uv pip install faiss-gpu-cu12
```

For ESMC, you can install the following packages and model weights:
```bash
uv pip uninstall transformers
uv pip install 'transformers<4.48.2' esm 'huggingface_hub[hf_transfer]'
HF_HUB_ENABLE_HF_TRANSFER=1 uv run --no-sync huggingface-cli download EvolutionaryScale/esmc-300m-2024-12
HF_HUB_ENABLE_HF_TRANSFER=1 uv run --no-sync huggingface-cli download EvolutionaryScale/esmc-600m-2024-12
```

For ESM2 with faesm, you can install the following package:
```bash
uv pip install transformers==4.48.1
uv pip install flash-attn --no-build-isolation
uv pip install 'faesm[flash_attn]'
```
Note: requires CUDA 11.7 or later.

Or, if you want to forego flash attention and just use SDPA
```bash
uv pip install faesm
```

### Building the datasets

The Pfam20 benchmark dataset can be built using the following command:
```bash
uv run python -m protein_search_evals.datasets.pfam
```

The Radical SAM benchmark dataset can be built using the following command:
```bash
tar -zxvf data/radicalsam.tar.gz -C data
uv run python -m protein_search_evals.datasets.radicalsam
```

### Running the embedding computation

To compute the embeddings for the Pfam20 dataset using ESM2-3B with faesm, run the following command:
```bash
nohup uv run python -m protein_search_evals.distributed_embeddings --config examples/pfam/embedding_configs/esm2-3B-faesm.yaml &> nohup.log &
```

Modify the YAML file to use different models or datasets.

### Computing embeddings on Polaris

Create a new conda environment with the following commands:
```bash
qsub -I -l select=1 -l filesystems=home:eagle -l walltime=1:00:00 -q debug -A FoundEpidem
module use /soft/modulefiles; module load conda
conda create -n genslm-embeddings python=3.12 -y
conda activate genslm-embeddings
```

Then install the package and dependencies:
```bash
git clone git@github.com:ramanathanlab/genslm-embeddings.git
cd genslm-embeddings
uv sync --locked
uv pip install flash-attn --no-build-isolation
uv pip install 'faesm[flash_attn]' faiss-gpu-cu12
```

Then run the embedding computation for SwissProt:
```bash
qsub examples/swissprot/submit.sh
```

To run the embedding computation for TrEMBL:
```bash
qsub examples/trembl/submit.sh
```

See the `examples` swissprot and trembl directories for more configuration details.


### Merging embeddings
To combine embeddings from multiple workflow runs, you can use symlinks:
```bash
SRC_DIR=/path/to/sprot-embeddings/esm3-3B_faesm_embeddings/embeddings
DST_DIR=/path/to/combined_embeddings

mkdir -p "$DST_DIR"
for dir in "$SRC_DIR"/*; do
    ln -s "$(realpath "$dir")" "$DST_DIR/$(basename "$dir")"
done
```
Simply replace the `SRC_DIR` and `DST_DIR` with the paths to the embeddings you want to combine.
You can run the command for multiple SRC_DIRs to merge embeddings from multiple runs.

Once you have all the embeddings in the same directory, you can run the following command to merge
them into a single Arrow file:
```bash
uv run protein_search_evals merge --dataset_dir /path/to/combined_embeddings/ --output_dir /path/to/combined_embeddings.merge
```

## Contributing

uv creates the `.venv` environment and installs the project in editable mode.
Synchronize the locked development and documentation dependencies, then install
the pre-commit hooks:

```bash
uv sync --locked --extra dev --group docs
uv run pre-commit install
```

Run quality checks and tests through the locked environment:

```bash
uv run pre-commit run --all-files
uv run pytest
```

### Documentation

The documentation site is built with ProperDocs. Install the documentation
dependencies and run a strict local build with:

```bash
uv sync --locked --only-group docs
uv run --no-sync properdocs build --strict
```

For live preview while editing documentation, run
`uv run --no-sync properdocs serve`.
