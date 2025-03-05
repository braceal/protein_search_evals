#!/bin/bash

# Define input data and output report directories
DATASET_DIR="data/radicalsam/"
REPORTS_DIR="examples/radicalsam/evaluations/reports"

# Define model names
MODELS=(
    "facebook/esm2_t6_8M_UR50D"
    "facebook/esm2_t12_35M_UR50D"
    "facebook/esm2_t30_150M_UR50D"
    "facebook/esm2_t33_650M_UR50D"
    "facebook/esm2_t36_3B_UR50D"
    # "facebook/esm2_t48_15B_UR50D"
    # "esmc_300m"
    # "esmc_600m"
)

# Define model directories containing the embeddings
MODEL_DIRS=(
    "examples/radicalsam/embeddings/all/esm2-8M_radicalsam_all"
    "examples/radicalsam/embeddings/all/esm2-35M_radicalsam_all"
    "examples/radicalsam/embeddings/all/esm2-150M_radicalsam_all"
    "examples/radicalsam/embeddings/all/esm2-650M_radicalsam_all"
    "examples/radicalsam/embeddings/all/esm2-3B_radicalsam_all"
    # "examples/radicalsam/embeddings/all/esm2-15B_radicalsam_all"
    # "examples/radicalsam/embeddings/all/esmc-300M_radicalsam_all"
    # "examples/radicalsam/embeddings/all/esmc-600M_radicalsam_all"
)

# Define precision types
PRECISIONS=("float32" "ubinary")

# Define dataset partitions
PARTITIONS=(
    "all"
    #"mega2"
    #"mega3"
    #"mega4"
    #"mega5"
)

# Ensure the reports directory exists
mkdir -p "$REPORTS_DIR"

# Loop over models, model directories, precision types, and dataset partitions
for i in "${!MODELS[@]}"; do
    MODEL_NAME="${MODELS[i]}"
    MODEL_DIR="${MODEL_DIRS[i]}"
    for PRECISION in "${PRECISIONS[@]}"; do
        for PARTITION in "${PARTITIONS[@]}"; do
            REPORT_NAME="$REPORTS_DIR/report_$(basename "$MODEL_NAME")_${PRECISION}_${PARTITION}"
            echo "Running evaluation with model: $MODEL_NAME, model_dir: $MODEL_DIR, precision: $PRECISION, partition: $PARTITION"
            python -m protein_search_evals.evaluate \
                --report_name "$REPORT_NAME" \
                --dataset_dir "$DATASET_DIR" \
                --dataset_partition "$PARTITION" \
                --model_dir "$MODEL_DIR" \
                --model_name "$MODEL_NAME" \
                --precision "$PRECISION"
        done
    done
done
