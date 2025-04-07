#!/bin/bash -l
#PBS -l select=2:system=polaris
#PBS -l place=scatter
#PBS -l walltime=0:60:00
#PBS -l filesystems=home:eagle
#PBS -j oe
#PBS -q debug
#PBS -A FoundEpidem

#------------------------------------------------------
# NOTE: Changing number of nodes also requires updating the YAML config file.


# Load the required environment
source ~/.bashrc

# Load the required modules
module use /soft/modulefiles; module load conda; conda activate protein_search_evals_03_25

# Set the environment variables
export HF_HOME=/lus/eagle/projects/CVD-Mol-AI/braceal/.cache

# Change to working directory
PROJECT_DIR=/lus/eagle/projects/FoundEpidem/braceal/projects/kbase-protein-search/src/protein_search_evals
cd $PROJECT_DIR

# Get the config file for this example
CONFIG_FILE=$PROJECT_DIR/examples/swissprot/esm2-3B-faesm.yaml

# Run the workflow
python -m protein_search_evals.distributed_embeddings --config $CONFIG_FILE
