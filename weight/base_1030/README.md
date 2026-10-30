---
language: en
tags:
- d2-omni
- experimental
- unfinished
- streaming
- audio-video-to-audio
- full-duplex
base_model:
- Qwen/Qwen3-Omni-30B-A3B-Instruct
- penguinfish1688/dialogue-to-duplex
---
# D2-Omni base_1030 — EXPERIMENTAL / UNFINISHED

This public research release stores the **base D2 weights plus our fine-tuned
components** that produced the team's current step-9,000 demos. It is unfinished
and intended for team reproduction. `base_1030` is the release label, **not** the
training update number.

[Hugging Face weights](https://huggingface.co/penguinfish1688/D2-Omni-base_1030-experimental)
· [GitHub code](https://github.com/penguinfish1688/6.4610-group-project)
· [Recorded demos](https://github.com/penguinfish1688/6.4610-group-project/tree/main/demo/checkpoint-9000)
· [Architecture video](https://github.com/penguinfish1688/6.4610-group-project/tree/main/demo/architecture)

The download destination is **`weight/base_1030/` under the repository root**.
On the original SLS checkout this is
`/data/sls/scratch/chang168/D2-Omni/weight/base_1030/`.
Large weights are stored on Hugging Face; GitHub tracks these instructions and
the download helper.

## Contents

```text
weight/base_1030/
  base_model/          Complete pinned Qwen3-Omni-30B-A3B-Instruct backbone
  base_d2/             Qwen D2 80-ms weights, causal audio encoder, release config
  finetuned/           Step-9,000 trained parameters in one FP32 safetensors file
  vision_encoder/      Frozen native Qwen vision tower
  examples/            The three recorded demo inputs, prompts, video and features
  reference_results/   Original generated text and full/trimmed output audio
  code/source.tar.gz   Matching source snapshot (commit recorded in release.json)
  inference-config.json
  release.json
  SHA256SUMS
```

The backbone is supplied in its original precision; the inference loader converts
frozen routed experts to D2's native INT8 format. `finetuned/model.safetensors`
contains all 1,347 trained parameters, including Thinker/Talker adapters, audio
frontend/control parameters and the added vision embeddings. No optimizer or
training RNG state is needed for inference. The original four-rank training
checkpoint remains under `runs/d2-omni-xl-80ms/` on SLS.

## Download

Clone the code and create a Python 3.12 environment:

```bash
git clone https://github.com/penguinfish1688/6.4610-group-project.git D2-Omni
cd D2-Omni
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install 'huggingface-hub>=1.5,<2'
python -m scripts.download_release
```

This downloads the public snapshot pinned in `hub.json` into `weight/base_1030/`.
No Hugging Face token is required. Allow about **75 GB** for the release, plus
space for the Python/CUDA environment and generated outputs. Downloads resume
when rerun. For a full additional local integrity pass:

```bash
python -m scripts.download_release --verify
```

## Reproduce the three video examples

Use Linux, Python 3.12 and a 48-GB NVIDIA GPU. We recommend 180 GB of host RAM;
the fresh reproduction run peaked near 123 GiB of process RSS during loading.
The SLS inference job requests one A6000, 180 GB RAM and eight CPU cores. The
verified runtime uses PyTorch 2.11.0 with CUDA 12.6, Transformers 5.13.0 and
BF16 compute. GPU residency is approximately 36 GiB after INT8 conversion.

In that environment, install the matching CUDA wheels before the packages:

```bash
source .venv/bin/activate
python -m pip install torch==2.11.0 torchaudio==2.11.0 torchvision==0.26.0 \
  --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r weight/base_1030/requirements-inference.txt
python -m pip install --no-deps -e model/backbone -e .
python -m scripts.infer_omni --bundle weight/base_1030 \
  --output runs/reproduce-base_1030
```

On the existing SLS installation, use its read-only runtime instead:

```bash
sbatch scripts/infer_omni.sbatch --bundle weight/base_1030 \
  --output runs/reproduce-base_1030
```

Each case produces `input.wav`, full-timeline `output.wav`, `output.txt`, a frame
trace and a `result.json`; the packaging command below adds trimmed `response.wav`. Open
`runs/reproduce-base_1030/index.html` to review them. The released source snapshot
can also be extracted into a new directory; it includes the same inference code.
Generation seeds and sampling settings are stored in `inference-config.json`.
Hardware or kernel changes may affect sampling, so compare content and speech
completion as well as exact bytes.

To render the same synchronized demo-video layout, install FFmpeg and DejaVu
fonts, then run:

```bash
python -m scripts.package_inference --run runs/reproduce-base_1030
python -m scripts.render_inference_demo --run runs/reproduce-base_1030 \
  --placeholder weight/base_1030/examples/placeholder.png
```

The three MP4s are written under `runs/reproduce-base_1030/demo/`, each with
the source video/placeholder, input soundtrack, generated audio and text.

The three inputs contain the exact aligned, frozen visual features used by the
recorded runs, so reproduction does not need private SLS datasets. AVQA also
includes its actual video and input soundtrack; InstructS2S includes the fixed
placeholder image. The frozen vision encoder is included for preparing new
inputs with the repository's normal video pipeline. No speech or text targets
are fed back into the released inference loop: self-audio comes from the
model's generated waveform.

## Architecture and training

Every 80-ms tick streams user audio, delayed generated self-audio, a visual token
and a text position through the Thinker. Its hidden states condition the native
Talker and waveform renderer. Text can finish before speech; inference continues
through the full recorded timeline.

Video is sampled at 1 fps and resized to 96×128 pixels (height×width), giving
12 merged tokens per image plus native DeepStack features. Each two seconds
uses 24 visual tokens and one visual padding slot across 25 audio ticks.
InstructS2S uses the encoded fixed random placeholder. AVQA uses real video
aligned with its scene audio. The vision encoder remains frozen.

Training starts from Qwen D2 80 ms and uses only InstructS2S and AVQA, with
same-task concatenation to at least 720 seconds, 90-second Transformer-XL chunks
and 90 seconds of detached memory. The official AVQA test split is excluded
from training. Native RMS +6-frame control placement is retained. After update
1,000, speech-code EOS CE weight is 5×; text RESPONSE/INTERRUPT remains 20×.

The original task system prompts are preserved exactly:

- InstructS2S: “You are a helpful spoken conversational assistant. Respond naturally when the user finishes speaking.”
- AVQA: “You are answering a multiple-choice audio question. Respond with the correct choice letter followed by the corresponding option wording.”

## Current results and limitations

These are three development examples, not an official test-set evaluation:

| Case | Current output | Observation |
| --- | --- | --- |
| AVQA 1, real video | `C. hail` | Matches the reference |
| InstructS2S, placeholder | Three tips on diet, exercise and sleep | Full answer is spoken |
| AVQA 2, real video | `C. street` | Incorrect; reference is `A. On the road` |

This checkpoint has incomplete evaluation and no demonstrated real-time
performance. The original eager inference runs took approximately 4–5 times
the input audio duration. Visual reliance, general speech completion and
robustness still need broader evaluation.

## Sources and attribution

- [Qwen3-Omni-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct), revision `26291f793822fb6be9555850f06dfe95f2d7e695`.
- [Dialogue-to-Duplex](https://github.com/penguinfish1688/D2-dialogue-to-duplex) and its [Qwen D2 release](https://huggingface.co/penguinfish1688/dialogue-to-duplex).
- Native INT8 runtime provenance is recorded in `model/INT8_UPSTREAM.json` in the code snapshot.
- Upstream model cards are retained under `base_model/README.md` and `base_d2/README.upstream.md`; upstream model/code terms continue to apply.
- The small example inputs come from the InstructS2S and AVQA material used in the recorded development demos. This package is not a redistribution of either full dataset.
