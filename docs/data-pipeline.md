# Build D2-Omni training data from new audio

`scripts/prepare_audio_dataset.py` turns paired user/assistant audio and assistant
transcripts into `d2_omni.prepared.xl.v1` files and a checksummed portable manifest.
It runs from this repository, with no private SLS data loaders. The historical
checkpoint-9,000 corpus is available at
[Hugging Face](https://huggingface.co/datasets/penguinfish1688/D2-Omni-step9000-training-data).
Its [training instructions](../data/releases/step9000/TRAINING.md) cover setup
and the original two-phase run.

## Input files

Create `my_data/conversations.jsonl`, one JSON object per line:

```json
{"id":"session-001","group_id":"recording-001","split":"train","turns":[{"user_audio":"audio/question-001.wav","assistant_audio":"audio/answer-001.wav","assistant_text":"The sound is rain falling on the roof."}]}
{"id":"session-002","group_id":"recording-002","split":"probe","turns":[{"user_audio":"audio/question-002.flac","assistant_audio":"audio/answer-002.flac","assistant_text":"You can hear a dog barking."}]}
```

Paths are relative to the input JSONL. Supply mono or stereo WAV/FLAC files at
their original sample rate; the builder averages channels and resamples to
24 kHz. `assistant_text` must transcribe the assistant audio, with no chat
template or special tokens. Use `turns` for multiple consecutive exchanges.
All fields shown above are required. This example shows the schema; two short
examples alone are insufficient for the four-GPU training recipe.

`group_id` identifies the original recording or conversation before splitting.
Keep every derivative of it in one split. The builder rejects shared group IDs
and byte-identical audio files across train/probe, duplicate episode IDs, missing
files, silent/nonfinite audio, unsupported fields and transcripts that cannot
fit the available 80-ms text slots. Different encodings/crops of the same
recording must still share `group_id`; hashes cannot detect all semantic overlap.

This interface constructs sequential, non-overlapping conversations. If you
start with mixed recordings, separate user/assistant turns and provide corrected
assistant transcripts first. It does not perform ASR, diarization, TTS, forced
alignment, or synthesize barge-in examples. There is no user-video input in this
audio builder; it uses the original fixed visual placeholder. The historical
AVQA preparation code remains in `scripts/prepare_avqa.py`.

## Install and run

Use Python 3.12 and the dependency installation in the training guide. Download
the model bundle with `python -m scripts.download_release`. Preprocessing loads
only the processor, frozen visual encoder and Mimi, not the 30B language model.
Use a GPU for practical dataset preparation; `--device cpu` works for small
checks. The first run downloads `kyutai/mimi` at pinned revision
`89091b3e466eb6a9d11e537bf26b144f194978f7`, or supply a local directory with `--mimi`.

```bash
python -m scripts.prepare_audio_dataset \
  --input my_data/conversations.jsonl \
  --output data/processed/my-audio \
  --bundle weight/base_1030 \
  --device cuda
```

Use a new output directory for each build; the command refuses to overwrite one.
The initial pass validates files and plans groups. It then materializes one
group at a time. Long individual conversations remain intact. A failed build
can leave a partial directory; choose a new directory when retrying.

## What the builder does

1. Resamples each clip to 24 kHz and trims leading/trailing frames below −60 dBFS
   RMS on the 80-ms clock. Retains complete audible frames and rejects all-silence
   inputs. No random audio augmentation is applied.
2. Places INTERRUPT six frames after the first audible user frame and RESPONSE
   six frames after the last. Assistant audio starts the following frame.
   Lexical targets follow RESPONSE consecutively, then terminal PAD. Overlong
   transcripts fail instead of being silently truncated.
3. Packs whole episodes within each split until a conversation is at least
   720 seconds. A short remainder is appended to the previous group. It never
   crosses splits, repeats examples to inflate the dataset, or fills 12 minutes
   with synthetic silence. Ordinary inter-turn gaps are retained.
4. Encodes the **complete clean assistant timeline**, including its silence,
   with streaming [Mimi](https://huggingface.co/docs/transformers/model_doc/mimi).
   It carries both transformer and convolution caches across 65-frame blocks,
   uses the first 16 codebooks in native order, and resets caches only at a new
   prepared conversation. IDs remain 0–2047. At control boundaries, it inserts
   Talker BOS 2149 / EOS 2150 in codebook zero and uses canonical D2 silence
   elsewhere. It does not concatenate independently encoded per-turn codes.
5. Builds 16-kHz causal mel features for environment and assistant streams.
   The self stream includes the 371-sample Code2Wav playback delay; the model
   adds the separate interaction delay. The clean Mimi target has no playback
   delay. Default Talker speaker conditioning comes from the model.
6. Encodes the fixed seed-1337 placeholder image and three DeepStack feature
   levels with the frozen vision encoder. Adds the original conversation prompt.
7. Runs the model's text/control and codec episode validators, checks clocks and
   speech masks, then saves tensors, checksums, source hashes, timing events,
   the Mimi revision and a ready-to-edit training config.

Generated files:

```text
my-audio/
  prepared/train-000000.pt
  prepared/probe-....pt
  manifest.json
  provenance.json
  train-config.json
```

## Train on the generated data

The existing four-GPU trainer requires **at least four prepared conversations
per split, each at least 720 seconds**. Collect enough independent data for
train and probe; packing and long episodes can affect the final count. Check
`training_ready` and `conversation_counts` in `provenance.json`.

The generated config selects `custom_audio`, the portable manifest and the base
D2 checkpoint. It starts a new run at 5× speech EOS with a default 9,000-update
budget. Edit `max_updates`, output path and reporting intervals for your study.
This new-data run is distinct from reproducing the historical checkpoint.

```bash
torchrun --standalone --nnodes=1 --nproc-per-node=4 \
  --module scripts.train_omni \
  --config data/processed/my-audio/train-config.json
```

Run from the repository root on one four-A6000 node. For a small preprocessing
smoke check, `--minimum-seconds 1 --device cpu` permits short groups, but the
output will report `training_ready: false` and the 720-second trainer contract
will reject it. That option does not turn a short example into a valid XL corpus.

For exact checkpoint-9,000 reproduction, download the released frozen tensors
and use the phase-one/phase-two configs. The paired-audio builder shares their
Mimi/tensor conventions but intentionally has a documented, simpler dialogue
construction policy than the historical dynamic InstructS2S augmentation.
