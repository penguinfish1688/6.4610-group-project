---
language:
- en
pretty_name: D2-Omni checkpoint-9000 prepared training corpus
size_categories:
- n<1K
configs:
- config_name: metadata
  data_files:
  - split: train
    path: train.jsonl
  - split: validation
    path: probe.jsonl
tags:
- audio
- video
- multimodal
- full-duplex
- mimi
- d2-omni
---

# D2-Omni checkpoint-9,000 training data

The exact prepared tensors used by the experimental D2-Omni checkpoint at
update 9,000: **75 files, 6,972,228,669 bytes (6.97 GB)**. These contain cached
audio/vision features and supervision, ready for the four-GPU XL trainer.

| Source | Split | Prepared conversations | Duration | AVQA clips |
| --- | --- | ---: | ---: | ---: |
| InstructS2S dynamic | train | 63 | 46,648.96 s | — |
| InstructS2S dynamic | probe | 4 | 2,941.52 s | — |
| AVQA | train | 4 | 5,296.40 s | 192 |
| AVQA | probe | 4 | 3,538.08 s | 128 |

`probe` is development data selected from training-source media. It is not the
official AVQA test set. All clips appearing in official `val_qa.json` are
excluded from both released AVQA splits. Train/probe source clips are disjoint.
The original selection requested 64 InstructS2S train rows; row 13 was excluded
because its lexical targets exceeded their timing capacity. The released 63
rows match the run manifest exactly.

**Train the checkpoint recipe:** see [TRAINING.md](TRAINING.md), including the
20× → 5× speech-EOS loss change after step 1,000. Starting from the base D2
checkpoint, run 1,000 updates, resume the saved optimizer/scheduler/XL state,
and continue to update 9,000. The dataset preserves exact prepared bytes;
retraining is not a claim of bit-identical model weights.

**Create data from your own audio:** use the
[GitHub audio pipeline](https://github.com/penguinfish1688/6.4610-group-project/blob/main/docs/data-pipeline.md).
Supply separate user and assistant audio files plus assistant transcripts. The
pipeline resamples audio, lays out response/control timing, streams the clean
assistant waveform through **Mimi's first 16 codebooks at 24 kHz / 12.5 Hz**,
builds causal mel features and placeholder visual features, validates targets,
and writes the same prepared format. See [DATA_PIPELINE.md](https://huggingface.co/datasets/penguinfish1688/D2-Omni-step9000-training-data/blob/main/DATA_PIPELINE.md)
for a complete input example, build command and training command. In the GitHub
checkout, the same guide is `docs/data-pipeline.md`. An unlabeled
mixed recording alone does not supply speaker roles or target transcripts.

```bash
python -m scripts.prepare_audio_dataset \
  --input my_data/conversations.jsonl \
  --output data/processed/my-audio \
  --bundle weight/base_1030 --device cuda
```

## Download and inspect

```python
from huggingface_hub import snapshot_download
snapshot_download(
    "penguinfish1688/D2-Omni-step9000-training-data",
    repo_type="dataset",
    local_dir="data/processed/step9000-release",
)
```

Record the Hub revision for repeatable downloads. From the downloaded directory,
run `sha256sum -c SHA256SUMS`. `manifest.json` contains ordered, relative paths,
sizes and SHA256 checksums; retain its order because the trainer assigns rows to
ranks and cycles through them deterministically. `train.jsonl` and `probe.jsonl`
provide inspection metadata, tensor shapes, prompt IDs and source identities.
Use `torch.load(path, map_location="cpu", mmap=True, weights_only=True)` to
inspect a prepared file. These are custom PyTorch tensors, not a `datasets.Audio`
table; download the tensors with `snapshot_download`. The optional Hub
`metadata` config exposes the JSONL inspection rows only (`probe` is named
`validation` in the viewer).

## Tensor format

Each file is `d2_omni.prepared.xl.v1`, a dictionary with `ticks`, `source`, `split`,
`tensors`, and either a `vision_bank` (InstructS2S placeholder) or `vision_segments`
(AVQA). One tick is 80 ms. All conversations are at least 720 seconds.

| Tensor | Shape | Meaning |
| --- | --- | --- |
| `env_mel`, `self_mel` | `[1,128,8*T]` | Causal 16-kHz audio features |
| `text_target` | `[1,T]` | Lexical, RESPONSE, INTERRUPT and PAD IDs |
| `codec_target` | `[1,T,16]` | Native Mimi targets plus codec BOS/EOS |
| `frame_mask`, `assistant_mask` | `[1,T]` | Valid frames and physical speech activity |
| `interrupt_end_weight_class` | `[1,T]` | Native end-control weight classes |
| `speaker_ids` | `[1]` | Native Talker speaker token |
| `system_prompt_input_ids` | `[1,P]` | Task prompt inserted once per conversation |

Ordinary Mimi codes are 0–2047; codebook zero uses BOS 2149 and EOS 2150.
Non-speech positions use D2's canonical silence frame. Visual features include
three DeepStack levels, 12 tokens per 96×128 image at 1 fps, hidden width 2048.
This release has all tensors needed to train, not the complete upstream raw
audio/video collections. Four InstructS2S probe files also retain their input
waveforms, as in the original prepared files.

## Provenance and source terms

`provenance/` contains the original run manifest, data audit, audio recipe,
excluded-row report, launch configurations, checkpoint-9,000 shard identities,
optimizer rates and source snapshot hashes. Historical absolute paths are
provenance only; the portable manifest and configs do not need those paths.
`source.tar.gz` includes the code used for this release and its portable loader.

Sources are the locally prepared Kokoro/zero-trimmed/dynamic InstructS2S variant
and [AVQA (Yang et al., ACM MM 2022)](https://github.com/AlyssaYoung/AVQA).
[InstructS2S originates in LLaMA-Omni](https://github.com/ictnlp/LLaMA-Omni).
Upstream source data and model terms continue to apply; this packaging does not
assign a new blanket license to third-party content. The run provenance does
not record a single independently verified redistribution license for the
combined derived corpus. Consult upstream terms before further redistribution.

Model: [D2-Omni base_1030 experimental](https://huggingface.co/penguinfish1688/D2-Omni-base_1030-experimental).
Code: [6.4610-group-project](https://github.com/penguinfish1688/6.4610-group-project).
This is a small research/development corpus, not a benchmark score or the full
InstructS2S/AVQA datasets.
