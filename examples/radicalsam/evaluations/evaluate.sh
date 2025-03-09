#!/bin/bash

# Define input data and output report directories
DATASET_DIR="data/radicalsam/"
BASE_REPORTS_DIR="examples/radicalsam/evaluations/reports"

# Define model names
MODELS=(
    "facebook/esm2_t6_8M_UR50D"
    "facebook/esm2_t12_35M_UR50D"
    "facebook/esm2_t30_150M_UR50D"
    "facebook/esm2_t33_650M_UR50D"
    "facebook/esm2_t36_3B_UR50D"
    "esmc_300m"
    "esmc_600m"
)

# Define precision types
PRECISIONS=("float32" "ubinary")

# Function to run evaluations
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
                --precision "$PRECISION"
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
MODEL_DIRS_ALL=(
    "examples/radicalsam/embeddings/all/esm2-8M_radicalsam_all"
    "examples/radicalsam/embeddings/all/esm2-35M_radicalsam_all"
    "examples/radicalsam/embeddings/all/esm2-150M_radicalsam_all"
    "examples/radicalsam/embeddings/all/esm2-650M_radicalsam_all"
    "examples/radicalsam/embeddings/all/esm2-3B_radicalsam_all"
    "examples/radicalsam/embeddings/all/esmc-300M_radicalsam_all"
    "examples/radicalsam/embeddings/all/esmc-600M_radicalsam_all"
)

run_evaluation MODEL_DIRS_ALL[@] "all"

# Define model directories containing the embeddings for the mega2 partition
MODEL_DIRS_MEGA2=(
    "examples/radicalsam/embeddings/mega2/esm2-8M_radicalsam_mega2"
    "examples/radicalsam/embeddings/mega2/esm2-35M_radicalsam_mega2"
    "examples/radicalsam/embeddings/mega2/esm2-150M_radicalsam_mega2"
    "examples/radicalsam/embeddings/mega2/esm2-650M_radicalsam_mega2"
    "examples/radicalsam/embeddings/mega2/esm2-3B_radicalsam_mega2"
    "examples/radicalsam/embeddings/mega2/esmc-300M_radicalsam_mega2"
    "examples/radicalsam/embeddings/mega2/esmc-600M_radicalsam_mega2"
)

run_evaluation MODEL_DIRS_MEGA2[@] "mega2"

# Define model directories containing the embeddings for the mega3 partition
MODEL_DIRS_MEGA3=(
    "examples/radicalsam/embeddings/mega3/esm2-8M_radicalsam_mega3"
    "examples/radicalsam/embeddings/mega3/esm2-35M_radicalsam_mega3"
    "examples/radicalsam/embeddings/mega3/esm2-150M_radicalsam_mega3"
    "examples/radicalsam/embeddings/mega3/esm2-650M_radicalsam_mega3"
    "examples/radicalsam/embeddings/mega3/esm2-3B_radicalsam_mega3"
    "examples/radicalsam/embeddings/mega3/esmc-300M_radicalsam_mega3"
    "examples/radicalsam/embeddings/mega3/esmc-600M_radicalsam_mega3"
)

run_evaluation MODEL_DIRS_MEGA3[@] "mega3"

# Define model directories containing the embeddings for the mega4 partition
MODEL_DIRS_MEGA4=(
    "examples/radicalsam/embeddings/mega4/esm2-8M_radicalsam_mega4"
    "examples/radicalsam/embeddings/mega4/esm2-35M_radicalsam_mega4"
    "examples/radicalsam/embeddings/mega4/esm2-150M_radicalsam_mega4"
    "examples/radicalsam/embeddings/mega4/esm2-650M_radicalsam_mega4"
    "examples/radicalsam/embeddings/mega4/esm2-3B_radicalsam_mega4"
    "examples/radicalsam/embeddings/mega4/esmc-300M_radicalsam_mega4"
    "examples/radicalsam/embeddings/mega4/esmc-600M_radicalsam_mega4"
)

run_evaluation MODEL_DIRS_MEGA4[@] "mega4"

# Define model directories containing the embeddings for the mega5 partition
MODEL_DIRS_MEGA5=(
    "examples/radicalsam/embeddings/mega5/esm2-8M_radicalsam_mega5"
    "examples/radicalsam/embeddings/mega5/esm2-35M_radicalsam_mega5"
    "examples/radicalsam/embeddings/mega5/esm2-150M_radicalsam_mega5"
    "examples/radicalsam/embeddings/mega5/esm2-650M_radicalsam_mega5"
    "examples/radicalsam/embeddings/mega5/esm2-3B_radicalsam_mega5"
    "examples/radicalsam/embeddings/mega5/esmc-300M_radicalsam_mega5"
    "examples/radicalsam/embeddings/mega5/esmc-600M_radicalsam_mega5"
)

run_evaluation MODEL_DIRS_MEGA5[@] "mega5"
