# D2-Omni

NLP group project: extend [Dialogue-to-Duplex (D2)](https://github.com/penguinfish1688/D2-dialogue-to-duplex)
with streaming vision so an assistant can **see, listen, and speak simultaneously**.
We reuse Qwen3-Omni's pretrained visual understanding and D2's full-duplex speech
pipeline instead of training an omni model from scratch.

**Current status:** repository initialized, D2 source imported, and an editable
Qwen3-Omni vision encoder included. Streaming vision, distillation, audiovisual
training, and the webcam demo are research work still to implement.

## Repository layout

```text
data/
  stage1/                 Video preparation for vision distillation (CK)
  stage2/                 Synchronized audiovisual conversation preparation
  examples/               Small illustrative manifests, without media
model/
  backbone/               D2 source, tracked directly in this repository
    qwen3-omni-D2/         Main backbone to modify for vision integration
  vision_encoder/         Editable Qwen3-Omni vision tower and provenance
configs/                  Initial experiment design settings
scripts/                  Checkpoint preparation utilities
tests/                    Small CPU checks for the extracted vision encoder
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

Start with one streaming video-QA example as a Stage II smoke test before
attempting longer conversations. GRPO is a stretch goal after the core system works.

## Approach

**Stage I — streaming vision and compression.** First profile the native visual
tower on incremental frames. If its cost is acceptable, prioritize token
compression and timeline alignment. Otherwise, distill a streaming student from
the frozen native encoder. Candidate video sources include AVQA training videos,
MSR-VTT, and ActivityNet Captions; QA labels are unnecessary for representation
distillation. One vision token per 80-ms interval (12.5 tokens/s) is an initial
budget to evaluate, not an established architecture or measured result.

**Stage II — audiovisual full-duplex SFT.** Add a visual lane alongside user audio
and delayed assistant audio. Build timestamped conversations from video QA,
synthesize speech where needed, and include interruptions, overlap, and silence.
Mix in audio-only D2 examples and ordinary AVQA training examples. Initialize
from Qwen3-Omni-D2, initially freeze the speech decoder, and investigate selective
updates or LoRA. Preserve causal access to media and D2's training/inference order.

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
python -m pip install -c model/backbone/qwen3-omni-D2/configs/requirements.txt -e .
python -m unittest discover -s tests -v
```

The D2 runtime has additional GPU/driver and system dependency requirements; use
the imported [installation guide](model/backbone/README.md#installation) as the
source of truth. Run model inference and training on an allocated ORCD GPU, with
model/download caches on scratch or pool. Use `HF_HOME` to select the cache path.
Confirm current ORCD partitions, GPU limits, and storage quotas for your account
before planning runs; the project notes' quotas are planning assumptions.

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
to the original repository. It is a reference remote, not this project's publishing
destination. No D2-Omni GitHub remote has been created.

D2's optional benchmark dependencies remain pinned submodules, registered in
the root `.gitmodules`. Fetch them only when needed, from this directory:

```bash
git submodule update --init --recursive
```

The original D2 source is currently unchanged apart from its provenance file.
Its loader removes `qwen.thinker.visual`; the [integration notes](docs/architecture.md)
identify this and the additional changes needed for vision support.

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
