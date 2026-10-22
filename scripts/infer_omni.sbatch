#!/usr/bin/env bash
#SBATCH --job-name=d2-omni-inference
#SBATCH --partition=a6
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=1-00:00:00
#SBATCH --output=/data/sls/scratch/chang168/D2-Omni/logs/inference-%j.out
#SBATCH --error=/data/sls/scratch/chang168/D2-Omni/logs/inference-%j.err
set -euo pipefail
cd /data/sls/scratch/chang168/D2-Omni
export D2_OMNI_USE_GPU=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
srun --kill-on-bad-exit=1 bash scripts/sls_python.sh -m scripts.infer_omni "$@"
