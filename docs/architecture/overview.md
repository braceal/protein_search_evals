# Architecture Overview

GenSLM Embeddings separates model inference, persisted embeddings, search,
and evaluation so large embedding collections can be generated once and reused
across search experiments.

## Module map

| Area | Location | Responsibility |
| --- | --- | --- |
| Encoders | `genslm_embeddings/embed/encoders/` | Load protein language models and compute token representations |
| Pooling | `genslm_embeddings/embed/poolers.py` | Convert token representations into sequence embeddings |
| Storage | `genslm_embeddings/embed/writers.py` | Write and merge Hugging Face datasets |
| Distribution | `genslm_embeddings/distributed_embeddings.py` | Submit embedding work through Parsl |
| Compute backends | `genslm_embeddings/parsl.py` | Configure workstation, Swing, and Polaris execution |
| Search | `genslm_embeddings/search.py` | Quantize embeddings and build or query FAISS indexes |
| Evaluation | `genslm_embeddings/evaluate.py` | Compute sequence- and cluster-level retrieval accuracy |
| Datasets | `genslm_embeddings/datasets/` | Build and load benchmark datasets |

## Embedding pipeline

`distributed_embeddings.py` loads a YAML configuration into Pydantic models.
The selected encoder processes FASTA sequences, a pooler produces one vector
per sequence, and a writer stores each worker result as a Hugging Face dataset.
The CLI can merge the resulting shards without recomputing embeddings.

Encoders can optionally pool selected transformer blocks in a single forward
pass. Multi-layer outputs use `(sequence, layer, hidden)` order while sequence
and tag columns remain one-dimensional. Search resolves a transformer block
through the stored layer metadata and passes its two-dimensional slice to
FAISS.

## Search pipeline

`FaissIndex` loads stored embeddings, optionally quantizes them, and constructs
an exact, IVF, or HNSW index. `Retriever` embeds queries with the configured
encoder and searches the index for nearest-neighbor matches.

## Evaluation pipeline

Dataset classes expose sequences and their benchmark cluster assignments. The
evaluator removes self-hits, compares retrieved neighbors with the expected
cluster, and reports sequence-level and aggregate cluster-level accuracy.
