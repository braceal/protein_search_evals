#!/bin/bash

# Compute per-cluster distance variance (cosine similarity and Hamming distance)
# for each model's Pfam embeddings. No GPU required.

DATASET_DIR="data/pfam/"
BASE_REPORTS_DIR="examples/pfam/evaluations/pr_curve/reports-cluster-variance"
PARTITION="seed-42"

MODELS=(
    "facebook/esm2_t6_8M_UR50D"
    "facebook/esm2_t12_35M_UR50D"
    "facebook/esm2_t30_150M_UR50D"
    "facebook/esm2_t33_650M_UR50D"
    "facebook/esm2_t36_3B_UR50D"
    "facebook/esm2_t48_15B_UR50D"
    "esmc_300m"
    "esmc_600m"
    "Rostlab/prot_t5_xl_half_uniref50-enc"
)

MODEL_DIRS=(
    "examples/pfam/embeddings/esm2-8M_pfam20_seed-42"
    "examples/pfam/embeddings/esm2-35M_pfam20_seed-42"
    "examples/pfam/embeddings/esm2-150M_pfam20_seed-42"
    "examples/pfam/embeddings/esm2-650M_pfam20_seed-42"
    "examples/pfam/embeddings/esm2-3B_pfam20_seed-42"
    "examples/pfam/embeddings/esm2-15B_pfam20_seed-42"
    "examples/pfam/embeddings/esmc-300M_pfam20_seed-42"
    "examples/pfam/embeddings/esmc-600M_pfam20_seed-42"
    "examples/pfam/embeddings/prottrans_pfam20_seed-42"
)

REPORTS_DIR="${BASE_REPORTS_DIR}/${PARTITION}"
mkdir -p "$REPORTS_DIR"

for i in "${!MODELS[@]}"; do
    MODEL_NAME="${MODELS[i]}"
    MODEL_DIR="${MODEL_DIRS[i]}"
    OUTPUT_JSON="${REPORTS_DIR}/cluster_variance_$(basename "$MODEL_NAME")_${PARTITION}.json"

    echo "Running cluster variance: model=$MODEL_NAME, model_dir=$MODEL_DIR"
    python -m protein_search_evals.cluster_distance_variance \
        --output "$OUTPUT_JSON" \
        --dataset_dir "$DATASET_DIR" \
        --dataset_partition "$PARTITION" \
        --model_dir "$MODEL_DIR" \
        --model_name "$MODEL_NAME"
done

echo "Done. Reports in ${REPORTS_DIR}"
