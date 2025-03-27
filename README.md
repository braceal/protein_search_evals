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
pip install esm
pip install "huggingface_hub[hf_transfer]"
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download EvolutionaryScale/esmc-300m-2024-12
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download EvolutionaryScale/esmc-600m-2024-12
```

For ESM2 with faesm, you can install the following package:
```bash
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
