
#!/bin/bash

# ==============================================================================
# SLURM RESOURCE REQUESTS
# ==============================================================================

# A descriptive name for your job in the queue
#SBATCH --job-name="G8_Hadoop_Inference"

# Set to 4 hours to allow for model downloading and sequential 34B model inference
#SBATCH --time=04:00:00

# This requests 2x NVIDIA A100 GPUs (Required for Heavyweight models)
#SBATCH --gres=gpu:a100:2

# Specifies the queue/partition.
#SBATCH --partition=gpu

# 240G is sufficient for loading the 34B weights into CPU RAM initially.
#SBATCH --mem=240G

# Standard log file format for a single run
#SBATCH --output=llm_inference_g8_all.log

# ==============================================================================
# 1. ENVIRONMENT SETUP
# ==============================================================================

module purge
module load lang/Python/3.10.4-GCCcore-11.3.0
module load system/CUDA/12.4.0

# Update the path below with your specific project group ID.
PROJECT_PATH="/scratch/hpc-prf-dssecs/savani"

source $PROJECT_PATH/venv/bin/activate

# ==============================================================================
# 2. OPTIMIZATIONS & SECRETS
# ==============================================================================

# Centralized storage for model weights to avoid filling your home directory
export HF_HOME="$PROJECT_PATH/huggingface_cache"

# Memory optimization to help prevent "Out of Memory" (OOM) errors
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Your Hugging Face Access Token
export HF_TOKEN="MY TOKEN"

# ==============================================================================
# 3. EXECUTION
# ==============================================================================

echo "Starting Hadoop Client Core Analysis (All Clusters) on $(hostname)..."

# Ensure sample.py is in the same directory where you submit this .sh file
python sample.py

# --- MONITORING ---
# 1. Check if job is running: squeue -u $USER
# 2. View live logs: tail -f llm_inference_g8_all.log

