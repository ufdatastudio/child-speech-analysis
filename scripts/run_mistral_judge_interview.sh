#!/bin/bash
#SBATCH --account ufdatastudios
#SBATCH --job-name mistral-judge-interview
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --time=6:00:00
#SBATCH --mem=80GB
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
OUTPUT_DIR="${BASE_DIR}/results/Interview/MistralJudge"

# Use /orange for model caches to avoid home quota
export HF_HOME="${BASE_DIR}/.cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${BASE_DIR}/.cache/transformers"
mkdir -p "${HF_HOME}" "${HF_HUB_CACHE}" "${TRANSFORMERS_CACHE}"

# Performance knobs
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Optional: set HF token for gated models
# export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-}"

echo "===== Starting Mistral Judge Evaluation (Interview) ====="
echo "CSV Path: ${CSV_PATH}"
echo "Output Dir: ${OUTPUT_DIR}"

# Run Mistral judge evaluation for interview
python3 -c "
import sys
sys.path.append('${BASE_DIR}')
from models.multi_open_judge_int import *

# Override to use only Mistral
JUDGE_MODELS.clear()
JUDGE_MODELS['mistral'] = 'mistralai/Mistral-7B-Instruct-v0.3'

# Run evaluation
for jname, jid in JUDGE_MODELS.items():
    try:
        evaluate_with_judge(
            csv_path='${CSV_PATH}',
            judge_name=jname,
            judge_id=jid,
            model_columns=['AF3_summary', 'Qwen_summary', 'Kimi_summary', 'Salmon_summary', 'GAMA_summary'],
            reference_column='Llama_summary',
            output_dir='${OUTPUT_DIR}',
            max_new_tokens=512,
            temperature=0.2
        )
        torch.cuda.empty_cache()
        print(f'Completed {jname} evaluation')
    except Exception as e:
        print(f'Failed {jname} evaluation: {e}')
print('Mistral interview evaluation complete')
"

echo "===== Mistral Interview Evaluation completed ====="
