#!/bin/bash
#SBATCH --account ufdatastudios
#SBATCH --job-name llama-judge
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --time=8:00:00
#SBATCH --mem=64GB
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
CSV_PATH="${BASE_DIR}/results/Interview/interview_combined.enriched.csv"
OUTPUT_DIR="${BASE_DIR}/results/Interview/Evaluations"

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

echo "===== Starting LLM-as-a-Judge Evaluation ====="
echo "CSV Path: ${CSV_PATH}"
echo "Output Dir: ${OUTPUT_DIR}"

# Run the LlamaJudge evaluation
python "${BASE_DIR}/models/LlamaJudge.py" \
  --csv_path "${CSV_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --model_columns AF3_summary Qwen_summary Kimi_summary Salmon_summary \
  --reference_column Llama_summary \
  --model_id "meta-llama/Meta-Llama-3.1-8B-Instruct" \
  --max_new_tokens 512 \
  --temperature 0.2

echo "===== Evaluation completed ====="
