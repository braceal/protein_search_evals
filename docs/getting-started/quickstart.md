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

## Merge embedding shards

Distributed jobs write one Hugging Face dataset per worker. Merge those shards
with the package CLI:

```bash
genslm-embeddings merge \
  --dataset_dir /path/to/embedding-shards \
  --output_dir /path/to/merged-embeddings
```

Use `genslm-embeddings --help` to see all CLI commands and options.
