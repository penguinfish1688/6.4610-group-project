# D2-Omni

NLP group project: extend [Dialogue-to-Duplex (D2)](https://github.com/penguinfish1688/D2-dialogue-to-duplex)
with streaming vision so an assistant can **see, listen, and speak simultaneously**.
We reuse Qwen3-Omni's pretrained visual understanding and D2's full-duplex speech
pipeline instead of training an omni model from scratch.

**Current status:** four-position audiovisual training runs on SLS with the
Qwen D2 80-ms checkpoint, real AVQA video, and InstructS2S placeholder images.
The initial run completed 1,000 updates, with fixed development loss changing
from 2.104 to 0.734 and a median step time of 8.08 seconds on four A6000 GPUs.
The continuation targets 10,000 total updates and reduces speech-code EOS CE
weight from 20 to 5, restoring model, optimizer, scheduler, RNG, and XL state.
The vision encoder remains frozen; this recipe uses resizing without distillation.

Initial free-running inference at the 20×-EOS checkpoint completed three
development examples: one of two AVQA answers matched its reference, and
InstructS2S produced sensible full text but speech ended after the introduction.
The speech cutoff remains unresolved; the EOS ×5 continuation needs evaluation.

## Repository layout

```text
data/
  stage1/                 Native vision resizing and feature preparation (CK)
  stage2/                 Synchronized audiovisual conversation preparation
  examples/               Small illustrative manifests, without media
model/
  backbone/               D2 source, tracked directly in this repository
    qwen3-omni-D2/         Main backbone to modify for vision integration
  vision_encoder/         Editable Qwen3-Omni vision tower and provenance
configs/                  Vision settings and SLS training recipe
scripts/                  Dataset preparation, XL training, checkpoint utilities
tests/                    Vision, four-position layout and XL recurrence checks
evaluation/               Benchmark protocol and planned evaluation work
demo/                     Planned webcam/microphone/speech demo
docs/                     Architecture, task ownership, and progress updates
```

Dataset media, cached features, model weights, and run outputs are ignored by
Git. Put them in `data/raw/`, `data/processed/`, `data/features/`, `checkpoints/`,
or an external scratch/pool directory.

## Minimum viable result

1. A model supporting simultaneous audiovisual input and speech output.
2. Held-out AVQA results demonstrating use of visual information.
3. A short working webcam conversation demo.
4. Measured latency and comparable baseline results.

The current implementation supports audiovisual training and short inference
examples. Full speech completion, visual ablations, official test evaluation,
and the live demo remain to be completed. GRPO is a later stretch goal.

## Approach

**Vision frontend.** Sample video at 1 fps and resize each independent image to
96x128 pixels (height x width). The native 16-pixel patches and 2x2 spatial merger
produce 12 tokens per image. Keep the encoder frozen and retain its DeepStack
features. Distribute 24 visual tokens plus one visual padding slot over 25 native
80-ms ticks. Distillation and a learned compressor are unnecessary for this first
version.

**Audiovisual full-duplex SFT.** Start from the Qwen D2 80-ms checkpoint and use
the native D2 training groups, control signals, and separate conversation
and AVQA system prompts, with an explicit speech EOS loss override. Train on
InstructS2S with an encoded fixed random placeholder image and AVQA with real
video aligned to its scene audio. These are
the only two sources. Shared LSKV audio datasets are read-only; derived data and
caches belong to D2-Omni. Use the official AVQA train/test split, including keeping
`val_qa.json` out of training. See [the SLS run config](configs/train-sls.json).

The current SLS recipe uses RMS +6-frame control placement, conversations of at
least 720 seconds, and Transformer-XL training with 90-second chunks and
90 seconds of detached KV memory. Frozen routed experts use D2's INT8 format;
selected expert weights and the remaining decoder parameters are sharded across
four A6000 GPUs. Full chunks use native lower-right causal FlashAttention;
padded chunks retain the explicit D2 mask.

The initial working corpus contains 63 InstructS2S training conversations and
192 real-video AVQA training clips. Separate development conversations come
from training-source media; official test media is excluded. The prepared data
audit and exact source manifest are stored with each run under `runs/`.

On SLS, submit and inspect the run with:

```bash
sbatch scripts/train_omni.sbatch
python scripts/training_status.py
```

Training was stopped for inference after update 1,000, then submitted for
continuation with 5× speech-code EOS weighting. Text RESPONSE/INTERRUPT weighting
remains 20×. To run held-out examples with their original training prompts and
generated self-audio feedback:

```bash
sbatch scripts/infer_omni.sbatch
```

The inference script restores all four checkpoint shards, keeps the waveform
renderer, and runs two real-video AVQA examples plus one InstructS2S example
with its encoded placeholder. Outputs include input/output WAVs, generated text,
frame traces, and a local HTML comparison under `runs/inference-step1000/`.
The default output path names the initial experiment; choose a new `--output`
directory when evaluating later checkpoints to preserve those results.
After inference, independently transcribe the speech and package short clips:

```bash
bash scripts/sls_python.sh -m scripts.transcribe_inference
python -m scripts.package_inference
```

`scripts/sls_python.sh` uses the shared read-only CUDA 12.6 runtime inside its
container. Checkpoints resume automatically from the run's `latest.json`.
The total step limit can be increased on resume; validation and checkpoint
intervals and the explicit speech-code EOS loss weight can also change. Other
training settings and source selection must match the checkpoint. The longer
run validates every 100 updates, reports each source separately, and saves every 250,
at shutdown, and at its final update. It retains D2's native learning-rate
schedule, which reaches its minimum rates at update 2,000.

The native audio clock is 80 ms; this does **not** guarantee 80-ms response
latency. Vision preprocessing, frame arrival, token count, model compute, and
audio playback all contribute to the measured delay.

See [architecture notes](docs/architecture.md), [data contracts](data/README.md),
and the [task board](docs/tasks.md).

## Setup

Use Python 3.12 and the dependencies pinned by the imported D2 revision. From
this directory:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -c model/backbone/qwen3-omni-D2/configs/requirements.txt -e './model/backbone[qwen]'
python -m pip install -c model/backbone/qwen3-omni-D2/configs/requirements.txt -e '.[data,dev]'
python -m pytest -q tests
python -m ruff check .
```

The D2 runtime has additional GPU/driver and system dependency requirements; use
the imported [installation guide](model/backbone/README.md#installation) as the
source of truth. Run model inference and training on an allocated ORCD GPU, with
model/download caches on scratch or pool. Use `HF_HOME` to select the cache path.
Confirm current ORCD partitions, GPU limits, and storage quotas for your account
before planning runs; the project notes' quotas are planning assumptions.

On the existing SLS installation, `bash scripts/sls_python.sh -m pytest -q tests`
uses the shared read-only runtime. The wrapper and Slurm files contain the SLS
paths and allocation settings; adapt those paths for another installation.
Independent speech transcription additionally requires `sherpa_onnx` and the
local ASR model paths configured in `scripts/transcribe_inference.py`.

To try the existing **audio-only** D2 baseline on a suitable GPU allocation:

```bash
d2-qwen app
```

This command may download the pretrained backbone and D2 weights. It does not
enable vision. Follow the backbone guide for remote port forwarding and inference.

For vision-only checkpoint extraction and a local encoder example, see
[model/vision_encoder](model/vision_encoder/README.md).

## Working on the backbone

`model/backbone` was cloned from D2 and imported as ordinary tracked source files.
Edit it directly and commit changes in this repository. It is not a separate
nested Git repository. The imported commit is recorded in
[UPSTREAM.json](model/backbone/UPSTREAM.json); the local `d2-upstream` remote points
to the original repository. This project's `origin` remote is
[penguinfish1688/6.4610-group-project](https://github.com/penguinfish1688/6.4610-group-project).

D2's optional benchmark dependencies remain pinned submodules, registered in
the root `.gitmodules`. Fetch them only when needed, from this directory:

```bash
git submodule update --init --recursive
```

The D2 source now includes the optional fourth visual position, DeepStack
injection, and matching cache accounting. The loader removes the original visual
tower; training prepares features with the separately loaded frozen encoder.
See the [integration notes](docs/architecture.md) for the current contract.

## Team

| Member | Contact | GitHub | Current ownership |
| --- | --- | --- | --- |
| Aidan Duncan | azd@mit.edu · 781-240-9418 | [Dunann](https://github.com/Dunann) | To be selected |
| Cheng Kuang Chang (CK) | chang168@mit.edu · 617-251-2024 | [penguinfish1688](https://github.com/penguinfish1688) | Stage I vision encoder and dataset |

Claim a task and add a short plan in [docs/tasks.md](docs/tasks.md). Record progress
with your name, date, artifacts, and next step in [docs/updates.md](docs/updates.md).

## Background

D2 first distills a causal audio encoder, then fine-tunes on aligned full-duplex
conversations. The [upstream repository](https://github.com/penguinfish1688/D2-dialogue-to-duplex)
reports 99.2% turn-taking success, 99.5% interruption success, about 0.4-second
interruption latency, and a 70.07 VoiceBench score for its Qwen3-Omni 80-ms model.
These are **D2 baseline results**, not D2-Omni results.

References: [D2 paper supplied by CK](https://drive.google.com/file/d/15YmwYVZDxiOqbLo6My9oOs07PegfrQU1/view),
[Qwen3-Omni](https://github.com/QwenLM/Qwen3-Omni),
[pinned vision source](https://github.com/huggingface/transformers/blob/v5.13.0/src/transformers/models/qwen3_omni_moe/modeling_qwen3_omni_moe.py).
The extracted vision code retains its Apache-2.0 license and attribution.
