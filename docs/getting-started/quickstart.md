# Quick Start

## Prepare a benchmark dataset

Build one of the included benchmark datasets:

```bash
python -m genslm_embeddings.datasets.pfam
```

For Radical SAM, extract the bundled source data first:

```bash
tar -xzf data/radicalsam.tar.gz -C data
python -m genslm_embeddings.datasets.radicalsam
```

## Compute embeddings

Embedding jobs are configured with YAML. The repository provides examples for
Pfam, Radical SAM, Swiss-Prot, and TrEMBL under `examples/`.

```yaml
input_dir: data/pfam/pfam20_seed-42
output_dir: examples/pfam/embeddings/esm2-150M
glob_patterns:
  - '*.fasta'

encoder_config:
  name: esm2
  pretrained_model_name_or_path: facebook/esm2_t30_150M_UR50D
  normalize_pooled_embeddings: true
  dataloader_batch_size: 8

compute_config:
  name: workstation
  available_accelerators: ['0']
```

Run the distributed embedding entry point with that configuration:

```bash
python -m genslm_embeddings.distributed_embeddings \
  --config examples/pfam/embedding_configs/esm2-150M.yaml
```

### Pool transformer layers

Set `pooled_layers` to `all` to pool every transformer block in one forward
pass, or provide zero-based block indices such as `[0, 5, 11]`:

```yaml
encoder_config:
  name: esm2
  pretrained_model_name_or_path: facebook/esm2_t6_8M_UR50D
  normalize_pooled_embeddings: true
  pooled_layers: all
```

Final-layer-only runs retain the existing `(sequences, hidden_dimension)`
embedding shape. Multi-layer runs store embeddings with shape
`(sequences, layers, hidden_dimension)` in the same Hugging Face dataset;
sequences and tags are stored only once. Dataset metadata records the
zero-based transformer block represented at each position on the layer axis.

When indexing or evaluating multi-layer embeddings, select a transformer block
explicitly:

```bash
python -m genslm_embeddings.evaluate \
  --model_dir /path/to/model-output \
  --model_name facebook/esm2_t6_8M_UR50D \
  --dataset_dir data/pfam \
  --dataset_partition seed-42 \
  --report_name esm2-layer-4 \
  --embedding_layer 4
```

Intermediate-layer pooling is supported by the Transformers implementations
of ESM-2 and ProtTrans. FAESM and ESM-Cambrian currently require
`pooled_layers: last` because their encoder adapters do not expose stable
intermediate hidden states. Requesting unsupported layers raises an error.

## Merge embedding shards

Distributed jobs write one Hugging Face dataset per worker. Merge those shards
with the package CLI:

```bash
genslm-embeddings merge \
  --dataset_dir /path/to/embedding-shards \
  --output_dir /path/to/merged-embeddings
```

Use `genslm-embeddings --help` to see all CLI commands and options.
