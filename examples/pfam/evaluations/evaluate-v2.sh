#!/bin/bash

# Define input data and output report directories
DATASET_DIR="data/pfam/"
BASE_REPORTS_DIR="examples/pfam/evaluations/reports-v2"

# Define model names
MODELS=(
    "facebook/esm2_t6_8M_UR50D"
    "facebook/esm2_t12_35M_UR50D"
    "facebook/esm2_t30_150M_UR50D"
    "facebook/esm2_t33_650M_UR50D"
    "facebook/esm2_t36_3B_UR50D"
    "esmc_300m"
    "esmc_600m"
    "Rostlab/prot_t5_xl_half_uniref50-enc"
)

# Define precision types
PRECISIONS=("float32" "ubinary")

# Function to run evaluations (requires 2 GPUs)
run_evaluation() {
    local model_dirs=("${!1}")
    local partition="$2"

    for i in "${!MODELS[@]}"; do
        MODEL_NAME="${MODELS[i]}"
        MODEL_DIR="${model_dirs[i]}"

        for PRECISION in "${PRECISIONS[@]}"; do
            REPORTS_DIR="${BASE_REPORTS_DIR}/${partition}"
            mkdir -p "$REPORTS_DIR"

            REPORT_NAME="${REPORTS_DIR}/report_$(basename "$MODEL_NAME")_${PRECISION}_${partition}"

            echo "Running evaluation with model: $MODEL_NAME, model_dir: $MODEL_DIR, precision: $PRECISION, partition: $partition"

            python -m protein_search_evals.evaluate \
                --report_name "$REPORT_NAME" \
                --dataset_dir "$DATASET_DIR" \
                --dataset_partition "$partition" \
                --model_dir "$MODEL_DIR" \
                --model_name "$MODEL_NAME" \
                --precision "$PRECISION" \
                --gpus 2
        done
    done
}

# Run each partition evaluation
# -----------------------------
# Note: The ESM-15B model is evaluated separately since it needs
# multiple GPUs to run the model inference plus the faiss search.
#
# Note: The model directories need to appear in the same order as
# the MODELS array defined above.

# Define model directories containing the embeddings for the all partition
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

run_evaluation MODEL_DIRS[@] "seed-42"
