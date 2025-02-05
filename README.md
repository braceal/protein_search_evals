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
