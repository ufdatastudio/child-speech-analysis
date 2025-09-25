#!/bin/bash
#SBATCH --account ufdatastudios
#SBATCH --job-name slp-summary
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --time=5:00:00
#SBATCH --mem=50GB
#SBATCH --mail-user=c.okocha@ufl.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition hpg-b200

set -euo pipefail

echo "===== GPU Info ====="
nvidia-smi || true

export CUDA_HOME=/apps/compilers/cuda/12.8.1
export PATH=$CUDA_HOME/bin:$PATH

# Paths
BASE_DIR="/orange/ufdatastudios/c.okocha/child__speech_analysis"
TRANSCRIPTS_DIR="${BASE_DIR}/Cws/transcript/Voices-CWS/interview"

# Use /orange for model caches to avoid home quota
export HF_HOME="${BASE_DIR}/.cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${BASE_DIR}/.cache/transformers"
mkdir -p "${HF_HOME}" "${HF_HUB_CACHE}" "${TRANSFORMERS_CACHE}"

# Performance knobs
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Optional: set HF token for gated models like Llama 3
# export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-}"



# Run the summarizer
python "${BASE_DIR}/models/Llama.py" \
  --input_dir "${TRANSCRIPTS_DIR}" \
  --model_id "meta-llama/Meta-Llama-3.1-8B-Instruct" \
  --max_new_tokens 4096 \
  --temperature 0.2

echo "Job completed."



