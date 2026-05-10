#!/bin/bash

# ==============================================================================
# SLURM RESOURCE REQUESTS
# =============================================================================


#SBATCH --job-name="hadoop_arch"
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=240G
#SBATCH --time=03:00:00
#SBATCH --qos=nocont
#SBATCH --output=hadoop_output_%j.log


# ===============================================================
# 1. ENVIRONMENT SETUP
# ==============================================================================

# Clears any modules loaded by default on the login node to prevent conflicts.
module purge 

# Loads the specific Python and CUDA modules.
module load lang/Python/3.10.4-GCCcore-11.3.0
module load system/CUDA/12.4.0

# Activates your virtual environment inside your ds4se folder.
# NOTE: This assumes your virtual environment is named "llm_env". 
# If it is named something else, change "llm_env" to match your folder name.
source $HOME/ds4se/llm_env/bin/activate


# ==============================================================================
# 2. OPTIMIZATIONS & SECRETS
# ==============================================================================

# Tells Hugging Face where to store/find model weights inside your project folder. 
export HF_HOME=$PC2PFS/hpc-prf-dssecs/$USER/huggingface_cache

# Memory optimization for PyTorch to prevent Out of Memory (OOM) errors.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Note: The HF_TOKEN is intentionally left out here for security.
# You will run `export HF_TOKEN=...` directly in your terminal before submitting.


# ==============================================================================
# 3. EXECUTION
# ==============================================================================

echo "Starting Model Inference..."

# Executes your Python script.
python sample.py

echo "Job Execution Complete."
