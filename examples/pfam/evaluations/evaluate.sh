#!/bin/bash

# Define the paths
DATASET_DIR="data/pfam/"
REPORTS_DIR="examples/pfam/evaluations/reports"
SCRIPT_PATH="examples/pfam/evaluations/evaluation.py"

# Ensure the reports directory exists
mkdir -p "$REPORTS_DIR"

# Define model names, model directories, and precision types
MODELS=(
    "facebook/esm2_t6_8M_UR50D"
    "facebook/esm2_t12_35M_UR50D"
    "facebook/esm2_t30_150M_UR50D"
    "facebook/esm2_t33_650M_UR50D"
    "facebook/esm2_t36_3B_UR50D"
    "facebook/esm2_t48_15B_UR50D"
    "esmc_300m"
    "esmc_600m"
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
)

PRECISIONS=("float32" "ubinary")

# Loop over models, model directories, and precision types
for i in "${!MODELS[@]}"; do
    MODEL_NAME="${MODELS[i]}"
    MODEL_DIR="${MODEL_DIRS[i]}"
    for PRECISION in "${PRECISIONS[@]}"; do
        REPORT_NAME="$REPORTS_DIR/report_$(basename "$MODEL_NAME")_${PRECISION}"
        echo "Running evaluation with model: $MODEL_NAME, model_dir: $MODEL_DIR, precision: $PRECISION"
        python "$SCRIPT_PATH" \
            --report_name "$REPORT_NAME" \
            --dataset_dir "$DATASET_DIR" \
            --model_dir "$MODEL_DIR" \
            --model_name "$MODEL_NAME" \
            --precision "$PRECISION"
    done
done
