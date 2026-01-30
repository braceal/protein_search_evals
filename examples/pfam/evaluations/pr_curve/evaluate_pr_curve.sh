#!/bin/bash

# Define input data and output report directories
DATASET_DIR="data/pfam/"
BASE_REPORTS_DIR="examples/pfam/evaluations/pr_curve/reports"

# Define model names
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

# Define precision types
PRECISIONS=("float32" "ubinary")

# Number of neighbors per query for building PR curve pairs
TOP_K=500

# Function to run PR curve evaluations (requires 2 GPUs)
run_pr_curve_evaluation() {
    local model_dirs=("${!1}")
    local partition="$2"

    for i in "${!MODELS[@]}"; do
        MODEL_NAME="${MODELS[i]}"
        MODEL_DIR="${model_dirs[i]}"

        for PRECISION in "${PRECISIONS[@]}"; do
            REPORTS_DIR="${BASE_REPORTS_DIR}/${partition}"
            mkdir -p "$REPORTS_DIR"

            OUTPUT_JSON="${REPORTS_DIR}/report_$(basename "$MODEL_NAME")_${PRECISION}_${partition}_pr_curve.json"

            echo "Running PR curve evaluation: model=$MODEL_NAME, model_dir=$MODEL_DIR, precision=$PRECISION, partition=$partition"

            python -m protein_search_evals.evaluate_pr_curve \
                --output "$OUTPUT_JSON" \
                --dataset_dir "$DATASET_DIR" \
                --dataset_partition "$partition" \
                --model_dir "$MODEL_DIR" \
                --model_name "$MODEL_NAME" \
                --precision "$PRECISION" \
                --top_k "$TOP_K" \
                --gpus 2
        done
    done
}

# Run each partition evaluation
# -----------------------------
# Model directories must appear in the same order as the MODELS array above.

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

run_pr_curve_evaluation MODEL_DIRS[@] "seed-42"
