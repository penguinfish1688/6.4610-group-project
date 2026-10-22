#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
shared=/data/sls/scratch/chang168/LSKV
mirror=/data/sls/scratch/chang168/LSKV-runtime
mkdir -p "$root/.cache/runtime_packages" "$root/.cache/tmp" "$root/.cache/torch_extensions"
for package in d2 d2_qwen; do
    target="$root/model/backbone/d2"
    if [[ "$package" == d2_qwen ]]; then target="$root/model/backbone/qwen3-omni-D2"; fi
    if [[ ! -e "$root/.cache/runtime_packages/$package" ]]; then
        ln -s "$target" "$root/.cache/runtime_packages/$package"
    fi
done
libraries="$shared/.venv-train/libshim"
for library in "$shared"/.venv-train/lib/python3.12/site-packages/nvidia/*/lib; do
    libraries="$libraries:$library"
done
gpu=()
if [[ ${D2_OMNI_USE_GPU:-0} == 1 ]]; then gpu=(--nv); fi
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export HF_HOME=/orcd/pool/006/chang168/D2-hf-release/cache/huggingface
export HF_HUB_CACHE="$HF_HOME/hub" HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TORCH_EXTENSIONS_DIR="$root/.cache/torch_extensions"
export TORCHINDUCTOR_CACHE_DIR="$root/.cache/inductor" TRITON_CACHE_DIR="$root/.cache/triton"
export XDG_CACHE_HOME="$root/.cache" TORCH_HOME="$root/.cache/torch"
unset LD_PRELOAD
export TMPDIR="${SLURM_TMPDIR:-/tmp}"
export PYTHONPATH="$root/.cache/runtime_packages:$root/model/backbone:$root"
exec "$shared/.sls/apptainer/bin/apptainer" exec "${gpu[@]}" \
    --bind "$root:$root" --bind "$shared:$shared:ro" --bind "$mirror/orcd:/orcd:ro" \
    --bind "$shared:/orcd/scratch/orcd/011/chang168/LSKV:ro" --pwd "$root" \
    --env "LD_LIBRARY_PATH=$libraries" \
    "$shared/.sls/containers/ubuntu24.sif" \
    /orcd/scratch/orcd/011/chang168/LSKV/.venv-train/bin/python "$@"
