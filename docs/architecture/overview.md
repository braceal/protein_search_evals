# Architecture Overview

Protein Search Evals separates model inference, persisted embeddings, search,
and evaluation so large embedding collections can be generated once and reused
across search experiments.

## Module map

| Area | Location | Responsibility |
| --- | --- | --- |
| Encoders | `protein_search_evals/embed/encoders/` | Load protein language models and compute token representations |
| Pooling | `protein_search_evals/embed/poolers.py` | Convert token representations into sequence embeddings |
| Storage | `protein_search_evals/embed/writers.py` | Write and merge Hugging Face datasets |
| Distribution | `protein_search_evals/distributed_embeddings.py` | Submit embedding work through Parsl |
| Compute backends | `protein_search_evals/parsl.py` | Configure workstation and Polaris execution |
| Search | `protein_search_evals/search.py` | Quantize embeddings and build or query FAISS indexes |
| Evaluation | `protein_search_evals/evaluate.py` | Compute sequence- and cluster-level retrieval accuracy |
| Datasets | `protein_search_evals/datasets/` | Build and load benchmark datasets |
| Reranking | `protein_search_evals/rerankers/` | Refine retrieved candidates with optional rerankers |

## Embedding pipeline

`distributed_embeddings.py` loads a YAML configuration into Pydantic models.
The selected encoder processes FASTA sequences, a pooler produces one vector
per sequence, and a writer stores each worker result as a Hugging Face dataset.
The CLI can merge the resulting shards without recomputing embeddings.

## Search pipeline

`FaissIndex` loads stored embeddings, optionally quantizes them, and constructs
an exact, IVF, or HNSW index. `Retriever` embeds queries with the configured
encoder, searches the index, and can pass candidates through a reranker.

## Evaluation pipeline

Dataset classes expose sequences and their benchmark cluster assignments. The
evaluator removes self-hits, compares retrieved neighbors with the expected
cluster, and reports sequence-level and aggregate cluster-level accuracy.
