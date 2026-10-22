# Stage II conversation data

Current recipe (2026-10-22): InstructS2S with encoded placeholder images, plus
AVQA with full source videos sampled at 1 fps and resized to 96x128 (12 tokens).
Use native D2's separate `conversation` and `avqa` system prompts and its original
signal/control configuration. The shared LSKV source data must remain read-only.
All converted samples, features, and manifests are written under D2-Omni.

Official AVQA splits: `train_qa.json` is train; `val_qa.json` is the official test
set (despite its filename). Development probes must be held out within training
sources. Download provenance: https://mn.cs.tsinghua.edu.cn/avqa/ and
https://github.com/AlyssaYoung/AVQA/tree/main/data/annotation.
The original SharePoint archive was downloaded successfully to
`data/raw/AVQA/AVQA.zip`. It contains annotations and precomputed HDF5 features,
not raw videos. The current run uses 384 raw MP4s acquired from
`juyil/AVQA-videos` at revision `e9538b7abd3dcf018f85c9e431d51f542728c13b`;
download receipts are in `data/raw/avqa/video-download-provenance.json`.
The prepared corpus selects 192 training clips and 128 development clips,
excluding every clip shared with the official test set.

This is a central project task because synchronized real-time audiovisual
interaction data is scarce.

1. Start with one training video and a question whose answer requires vision.
2. Record when the necessary visual evidence becomes available.
3. Assemble user questions and assistant answers with timestamps. Preserve the
   source video's environment audio where it matters for audiovisual QA.
4. Use TTS for missing speech and align the resulting audio to the timeline.
5. Concatenate QA interactions while recording each clip's timestamp mapping.
6. Add interruption, overlap, silence, and response-timing variants.
7. Mix AVQA with InstructS2S and its encoded placeholder image.
8. Convert records with `scripts/prepare_native_data.py` and
   `scripts/prepare_avqa.py`; train with `scripts/train_omni.py`.

The [example](../examples/stage2.jsonl) is an illustrative text-only QA
conversation before TTS, not a current training manifest.
Do not use held-out AVQA questions or their source videos for training.
