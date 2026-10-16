# Data

This directory holds preparation code, small manifests, and dataset documentation.
Store downloaded media in `raw/`, assembled conversations in `processed/`, and
cached teacher representations in `features/` (all ignored by Git).

## Stage I

Use training videos from AVQA and other suitable video datasets to distill or
compress the native visual representation. Preserve source video IDs, source
split, frame timestamps, preprocessing settings, and teacher revision. QA
annotations are optional. See [stage1/README.md](stage1/README.md).

## Stage II

Prepare a shared timeline with user video, user/environment audio, assistant
text, and optional assistant audio. Text-only answers can be synthesized later.
Keep scene audio and synthesized user speech identifiable before mixing them.
See [stage2/README.md](stage2/README.md) for the proposed construction pipeline.

The JSONL files in [examples/](examples/) illustrate the **draft v0 data contract**.
Their media paths are placeholders; they are not training samples. One line is
one source clip for Stage I or one assembled conversation for Stage II.

Conventions:

- All times are integer milliseconds on the conversation timeline. Spans are
  half-open `[start_ms, end_ms)`. Media paths are relative to the dataset root.
- For each clip, `source_start_ms` maps its original timestamp to `start_ms` on
  the assembled timeline. Keep original timestamps when concatenating videos.
- A visual observation may be consumed only after all its source frames have
  arrived. An 80-ms model tick does not imply the camera runs at 12.5 FPS.
- Media may be absent (`null` audio before TTS or empty video for audio-only
  replay); absence must not be represented by future frames or invented timing.
- Every QA turn has a task type, source ID, user text, assistant text, and
  timed visual evidence. Speech span timestamps are refined after TTS/alignment.
- Split by original video before concatenating or synthesizing. Keep all derived
  clips from a video in the same split, including Stage I data. Held-out AVQA
  examples must not enter either training stage.
- Track dataset version, source URLs, original splits, and redistribution terms.
  Upload reusable processed data/metadata to Hugging Face once the dataset is
  ready and source terms permit sharing; record the dataset ID and revision here.

The backbone currently accepts its own [prepared tensor manifest](../model/backbone/d2/SAMPLES.md).
These audiovisual JSONL records require a future conversion step. Adding a
`video` field alone does not make the existing D2 trainer consume visual inputs.
