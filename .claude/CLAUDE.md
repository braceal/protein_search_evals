# CLAUDE.md

This file provides guidance to coding agents working in this repository.

## Project Overview

Protein Search Evals benchmarks protein language model embeddings for sequence
search. It computes embeddings locally or with Parsl, stores them as Hugging
Face datasets, builds exact or approximate FAISS indexes, and evaluates
retrieval quality on protein-family datasets such as Pfam and Radical SAM.

## Commands

```bash
# Install
uv sync --locked --extra dev

# Lint and format
uv run pre-commit run --all-files
uv run ruff check .
uv run ruff format .

# Type check
uv run mypy protein_search_evals/

# Tests
uv run pytest

# Generate distributed embeddings from a YAML configuration
uv run python -m protein_search_evals.distributed_embeddings --config path/to/config.yaml

# Use the package CLI
uv run protein_search_evals --help
```

FAISS and model-specific acceleration packages are installed separately; see
`README.md` for the supported GPU and CPU installation options.

## Code Style

- **Formatter/Linter**: ruff (line length 79, single quotes, Python 3.8 target)
- **Docstrings**: NumPy convention
- **Imports**: `from __future__ import annotations` is required in every Python
  file; imports are kept one per line
- **Type checking**: mypy strict checks are enabled for production code
- Pre-commit hooks enforce ruff, mypy, codespell, trailing whitespace, and
  YAML/JSON validation

## Commit and PR Titles

Use [Conventional Commits](https://www.conventionalcommits.org/) for commit
messages and PR titles:

```text
<type>(optional-scope): <short imperative description>
```

Common types:

- `feat:` adds user-facing functionality and triggers a minor release.
- `fix:` corrects a bug and triggers a patch release.
- `perf:` improves performance and triggers a patch release.
- `docs:`, `test:`, `refactor:`, `build:`, `ci:`, `chore:`, and `style:` cover
  documentation, tests, internal code, tooling, automation, maintenance, and
  formatting changes.
- Append `!` after the type or scope (for example, `feat(search)!:`), or include
  a `BREAKING CHANGE:` footer, for a breaking change.

Keep the description concise, imperative, and lowercase; do not end it with a
period. Examples: `feat(search): add binary IVF support` and
`fix(embed): preserve sequence order during merge`.

PRs are squash-merged into `develop`, so the PR title becomes the commit that
Release Please evaluates. Every PR title must therefore follow this format,
even if its individual commits do not. Prefer Conventional Commit messages for
individual commits as well so they remain clear outside the squash-merge
history.

The package follows Semantic Versioning (`MAJOR.MINOR.PATCH`), starting from
`0.1.0`. Release Please derives version bumps from Conventional Commit types.

## Architecture

### Embedding pipeline

- `protein_search_evals/embed/encoders/` defines model-specific encoders and
  their Pydantic configurations.
- `protein_search_evals/embed/poolers.py` converts token representations into
  sequence-level embeddings.
- `protein_search_evals/embed/writers.py` persists and merges embedding outputs
  as Hugging Face datasets.
- `protein_search_evals/distributed_embeddings.py` loads a YAML configuration
  and distributes embedding jobs through Parsl.

### Search and evaluation

- `protein_search_evals/search.py` builds and queries FAISS indexes. It supports
  float32 and unsigned-binary embeddings with exact, IVF, and HNSW search.
- `protein_search_evals/evaluate.py` evaluates retrieval accuracy on Pfam and
  Radical SAM datasets.
- `protein_search_evals/evaluate_pr_curve.py` computes precision-recall results.
- `protein_search_evals/rerankers/` contains optional post-retrieval rerankers,
  including the PLM-BLAST implementation.

### Data and configuration

- `protein_search_evals/datasets/` builds and loads benchmark datasets.
- `protein_search_evals/parsl.py` contains local and HPC execution providers.
- `protein_search_evals/utils.py` provides the shared Pydantic configuration
  base and FASTA utilities.
- `examples/` contains benchmark notebooks, YAML configurations, and scheduler
  scripts. Large generated datasets and model outputs should not be committed.
