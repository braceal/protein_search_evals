# protein_search_evals
Protein search project

## Installation

To install the package, run the following command:
```bash
git clone git@github.com:braceal/protein_search_evals.git
cd protein_search_evals
pip install -U pip setuptools wheel
pip install -e .
```

To install Faiss, for GPU support with CUDA 12, run the following command:
```bash
pip install faiss-gpu-cu12
```

For ESMC, you can install the following packages and model weights:
```bash
pip uninstall transformers
pip install 'transformers<4.48.2'
pip install esm
pip install "huggingface_hub[hf_transfer]"
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download EvolutionaryScale/esmc-300m-2024-12
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download EvolutionaryScale/esmc-600m-2024-12
```

For ESM2 with faesm, you can install the following package:
```bash
pip install transformers==4.48.1
pip install flash-attn --no-build-isolation
pip install faesm[flash_attn]
```
Note: requires CUDA 11.7 or later.

Or, if you want to forego flash attention and just use SDPA
```bash
pip install faesm
```

### Building the datasets

The Pfam20 benchmark dataset can be built using the following command:
```bash
python -m protein_search_evals.datasets.pfam
```

The Radical SAM benchmark dataset can be built using the following command:
```bash
tar -zxvf data/radicalsam.tar.gz -C data
python -m protein_search_evals.datasets.radicalsam
```

### Running the embedding computation

To compute the embeddings for the Pfam20 dataset using ESM2-3B with faesm, run the following command:
```bash
nohup python -m protein_search_evals.distributed_embeddings --config examples/pfam/embedding_configs/esm2-3B-faesm.yaml &> nohup.log &
```

Modify the YAML file to use different models or datasets.

### Computing embeddings on Polaris

Create a new conda environment with the following commands:
```bash
qsub -I -l select=1 -l filesystems=home:eagle -l walltime=1:00:00 -q debug -A FoundEpidem
module use /soft/modulefiles; module load conda
conda create -n protein_search_evals_03_25 python=3.12 -y
conda activate protein_search_evals_03_25
```

Then install the package and dependencies:
```bash
git clone git@github.com:braceal/protein_search_evals.git
cd protein_search_evals
pip install -U pip setuptools wheel
pip install -e .
pip install flash-attn --no-build-isolation
pip install faesm[flash_attn]
pip install faiss-gpu-cu12
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
protein_search_evals merge --dataset_dir /path/to/combined_embeddings/ --output_dir /path/to/combined_embeddings.merge
```

## Contributing

For development, it is recommended to use a virtual environment. The following
commands will create a virtual environment, install the package in editable
mode, and install the pre-commit hooks.
```bash
python -m venv venv
source venv/bin/activate
pip install -U pip setuptools wheel
pip install -e '.[dev,docs]'
pre-commit install
```
To test the code, run the following command:
```bash
pre-commit run --all-files
tox -e py310
```
