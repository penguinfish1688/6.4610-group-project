#!/usr/bin/env bash
#SBATCH --job-name=d2-omni-80ms
#SBATCH --partition=a6
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=400G
#SBATCH --time=1-00:00:00
#SBATCH --signal=USR1@300
#SBATCH --output=/data/sls/scratch/chang168/D2-Omni/logs/slurm-%j.out
#SBATCH --error=/data/sls/scratch/chang168/D2-Omni/logs/slurm-%j.err
set -euo pipefail
cd /data/sls/scratch/chang168/D2-Omni
export D2_OMNI_USE_GPU=1
export NCCL_SOCKET_NTHREADS=2 NCCL_NSOCKS_PERTHREAD=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
srun --kill-on-bad-exit=1 bash scripts/sls_python.sh -m torch.distributed.run \
    --standalone --nnodes=1 --nproc-per-node=4 --module scripts.train_omni \
    --config configs/train-sls.json
