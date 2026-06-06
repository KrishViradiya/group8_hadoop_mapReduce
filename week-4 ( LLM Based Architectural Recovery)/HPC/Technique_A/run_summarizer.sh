#!/bin/bash
#SBATCH --job-name="phase-2"
#SBATCH --time=09:00:00
#SBATCH --gres=gpu:a100:2          # Bumping to 2x A100s to match Week 2 for the 34B model
#SBATCH --partition=gpu
#SBATCH --mem=240G                 # Matching Week 2 memory to prevent crashes
#SBATCH --output=week-4_output_log_%j.txt

# ==============================================================================
# 1. ENVIRONMENT SETUP (Copied exactly from your Week 2 success)
# ==============================================================================
module purge
module load lang/Python/3.10.4-GCCcore-11.3.0
module load system/CUDA/12.4.0

# Activate your virtual environment
source $HOME/ds4se/llm_env/bin/activate

# ==============================================================================
# 2. OPTIMIZATIONS & SECRETS
# ==============================================================================
# Force Hugging Face to use the lightning-fast scratch drive!
export HF_HOME=$PC2PFS/hpc-prf-dssecs/$USER/huggingface_cache

# PyTorch memory optimization to prevent OOM
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Add your HF Token here!


# ==============================================================================
# 3. EXECUTION
# ==============================================================================
echo "Starting week-4 File Summarization..."

python run_hierarchical_pipeline.py

echo "Summarization Loop Complete."