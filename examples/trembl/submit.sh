#!/bin/bash -l
#PBS -l select=127:system=polaris
#PBS -l place=scatter
#PBS -l walltime=24:00:00
#PBS -l filesystems=home:eagle
#PBS -j oe
#PBS -q prod
#PBS -A FoundEpidem

#------------------------------------------------------
# NOTE: Changing number of nodes also requires updating the YAML config file.


# Load the required environment
source ~/.bashrc

# Load the required modules
module use /soft/modulefiles; module load conda; conda activate genslm_embeddings_03_25

# Set the environment variables
export HF_HOME=/lus/eagle/projects/CVD-Mol-AI/braceal/.cache

# Change to working directory
PROJECT_DIR=/lus/eagle/projects/FoundEpidem/braceal/projects/kbase-protein-search/src/genslm_embeddings
cd $PROJECT_DIR

# Get the config file for this example
CONFIG_FILE=$PROJECT_DIR/examples/trembl/esm2-3B-faesm.yaml

# Run the workflow
python -m genslm_embeddings.distributed_embeddings --config $CONFIG_FILE
