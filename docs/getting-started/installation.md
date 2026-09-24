# Installation

## Base package

GenSLM Embeddings supports Python 3.10 and newer. Install uv, then
synchronize the locked environment:

```bash
git clone git@github.com:ramanathanlab/genslm-embeddings.git
cd genslm-embeddings
uv sync --locked
```

The distribution is named `genslm-embeddings`, its Python import package is
`genslm_embeddings`, and its command-line entry point is `genslm-embeddings`.

## FAISS

FAISS is installed separately because the appropriate package depends on the
target hardware. For a CUDA 12 environment:

```bash
uv pip install faiss-gpu-cu12
```

Use the CPU FAISS package when GPU search is not required.

## Model-specific acceleration

ESM-2 can use FAESM for faster inference:

```bash
uv pip install faesm
```

Flash Attention requires a compatible CUDA environment:

```bash
uv pip install flash-attn --no-build-isolation
uv pip install 'faesm[flash_attn]'
```

See the project `README.md` for ESM Cambrian model-download instructions and
cluster-specific setup notes.

## Development and documentation

Install the development and documentation dependencies, then enable the Git
hooks:

```bash
uv sync --locked --extra dev --group docs
uv run pre-commit install
```

Run the same lint and documentation checks used by CI:

```bash
uv run pre-commit run --all-files
uv run --no-sync properdocs build --strict
```
