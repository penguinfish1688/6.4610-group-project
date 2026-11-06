# Train the D2-Omni checkpoint-9,000 recipe

Use the exact prepared corpus in this dataset, the Qwen D2 **80-ms base** weights,
and the two supplied training configs. Do not initialize from the step-9,000
fine-tuned weights when reproducing this run from scratch.

## 1. Code and environment

Clone the [GitHub repository](https://github.com/penguinfish1688/6.4610-group-project).
For the frozen release code, extract this dataset's `source.tar.gz` in a new
directory and work inside its `D2-Omni/` folder. `provenance/source.json` records
the Git base commit and individual working-source hashes.

Use Linux, Python 3.12, **one node with four NVIDIA RTX A6000 48-GB GPUs** and
400 GB host memory, matching the original Slurm request. The current trainer
explicitly checks this GPU topology. Allow about 75 GB for the base release,
7 GB for this dataset, approximately 10 GB per distributed training checkpoint,
and space for the environment and temporary checkpoint publication.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.11.0 torchaudio==2.11.0 torchvision==0.26.0 \
  --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r weight/base_1030/requirements-inference.txt
python -m pip install --no-deps -e model/backbone -e .
python -m scripts.download_release
```

The original runtime was PyTorch 2.11.0/CUDA 12.6, Transformers 5.13.0 and BF16
compute. Native PyTorch SDPA provides the full-chunk causal attention kernel;
there is no separate `flash-attn` installation step.

## 2. Download and verify the data

Run from the code repository root:

```python
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="penguinfish1688/D2-Omni-step9000-training-data",
    repo_type="dataset",
    local_dir="data/processed/step9000-release",
)
```

Pin `revision` to the dataset commit for a repeatable download. Then:

```bash
(cd data/processed/step9000-release && sha256sum -c SHA256SUMS)
```

The configs use `prepared_manifest`, so the trainer reads these frozen tensors
directly without SLS, legacy source loaders, online TTS, or Mimi encoding at
training time. `portable_runtime: true` selects the vendored INT8 expert code.
Paths are relative to the code repository root. Keep them fixed between phases.

## 3. First 1,000 updates: 20× speech EOS

Start with a new `runs/retrain-step9000/` directory. Existing `latest.json` causes
automatic resume, so use a new output path for an independent experiment.

```bash
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
torchrun --standalone --nnodes=1 --nproc-per-node=4 \
  --module scripts.train_omni --config configs/train-step9000-phase1.json
```

## 4. Resume to update 9,000: 5× speech EOS

After phase one finishes successfully, run:

```bash
torchrun --standalone --nnodes=1 --nproc-per-node=4 \
  --module scripts.train_omni --config configs/train-step9000-phase2.json
```

Both configs use the same output directory. The trainer restores all rank
shards, optimizer, scheduler, RNG, sampling cursors, active conversation/chunk,
and detached XL state. The intended changes are `max_updates: 1000 → 9000` and
`talker_codec_eos_weight: 20 → 5`. This is **8,000 additional updates**.
Text RESPONSE/INTERRUPT weights remain 20×.

The historical continuation targeted 10,000, resumed again from update 2,250
after quota exhaustion, and was stopped at 9,033; its last complete retained
checkpoint is update 9,000. The release configs stop at 9,000 directly. They
save every 250 updates and retain the latest complete checkpoint. The original
run's launch records remain in `provenance/`.

## Training settings

| Setting | Value |
| --- | --- |
| Seed | 1337 |
| Clock / serialized layout | 80 ms; user audio, delayed self audio, visual, previous text |
| Current chunk | 1,125 ticks = 90 s = 4,500 stream positions |
| Detached XL memory | Previous 1,125 ticks / 90 s |
| Source schedule | One InstructS2S cohort then nine AVQA cohorts |
| Cohort | Four conversations, one per rank; complete all chunks before changing source |
| Optimizer | AdamW; betas (0.9, 0.999), epsilon 1e-8, weight decay 0 |
| Gradient clipping | Global norm 1.0 |
| LR schedule | 200-update warmup, cosine to minima at update 2,000; minima thereafter |
| Vision | Frozen, 1 fps, 96×128, 12 tokens/image, no distillation |
| Placeholder | Fixed random image, seed 1337, for InstructS2S |
| Control timing | RMS-based native placement, +6 frames |
| Frozen experts | D2 INT8, sharded final 8 Thinker and 20 Talker expert layers |
| Development | Every 100 updates; first two chunks of eight fixed probe conversations |

The 1:9 schedule is a **cohort schedule**, not exactly a 10%:90% ratio of updates
or hours, because conversation lengths differ. Learning rates are in
`provenance/optimizer.json`: Thinker/Talker/context/code-predictor LoRA start at
5e-5 and end at 1e-5; controls/codec heads start at 1e-4; codec embeddings at
5e-5; small modules at 2e-5; full audio frontend at 2e-6; audio LoRA at 1e-5.
Other groups end at 10% of their initial rate.

The prepared tensors are byte-identical to the original run inputs. Skipping
historical preparation can change RNG consumption before model initialization,
and kernels/runtime changes can affect numerical results. These instructions
reproduce the data and training recipe; the new four-GPU run has not been
executed as part of this release and bit-identical weights are not guaranteed.

For **new audio**, follow [DATA_PIPELINE.md](https://huggingface.co/datasets/penguinfish1688/D2-Omni-step9000-training-data/blob/main/DATA_PIPELINE.md). That builder
creates new paired-audio examples using the same tensor/codec contract. It does
not regenerate the historical dynamic augmentation choices or AVQA media join.
