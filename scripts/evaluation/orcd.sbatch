#!/usr/bin/env bash
#SBATCH --job-name=d2-omni-eval
#SBATCH --partition=mit_normal_gpu
#SBATCH --gpus=l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=02:00:00
#SBATCH --output=logs/eval-%j.out
#SBATCH --error=logs/eval-%j.err
set -euo pipefail
# Submit from the repository root after: mkdir -p logs
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
module load miniforge
source activate "${D2_OMNI_ENV:-$PWD/.venv}"
export HF_HOME="$PWD/.cache/huggingface-download"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
srun python -u -m scripts.evaluation "$@"
