# Progress updates

Use `YYYY-MM-DD — name`, followed by what changed, evidence/artifacts, and the
next step. Distinguish implemented behavior from planned experiments.

## 2026-10-09 — CK (from the initial project notes)

- Proposed extending D2 with streaming vision.
- Suggested running D2 on ORCD first, using a GPU with more than 70 GB VRAM.
- Invited teammates to clarify the writing and claim project tasks.

## 2026-10-16 — Codex, repository setup requested by CK

- Created the D2-Omni repository layout and imported D2 source at the revision
  recorded in `model/backbone/UPSTREAM.json`.
- Added editable native Qwen3-Omni vision code, a local checkpoint export utility,
  and small CPU tests against the pinned upstream implementation.
- Added draft timestamped data examples, research settings, task ownership,
  and evaluation/demo plans.
- Validation: tiny CPU models matched upstream outputs and parameter gradients
  with eager and SDPA attention. Vision-only checkpoint export/reload passed for
  single-file and sharded snapshots. All 117 imported D2 files matched upstream.
- Next: profile the native vision path and define the first Stage II sample.

## 2026-10-22 — Codex, first four-stream training implementation

- Added user-audio, self-audio, vision, and text positions to the native Qwen
  D2 80-ms model. Frozen native image encoding produces 12 tokens at 96×128;
  1-fps frames occupy 24 visual positions plus one padding position per 2 s.
- Added read-only InstructS2S ingestion with an encoded random placeholder and
  real AVQA videos, using D2's distinct original task prompts and RMS +6 timing.
  Training excludes every official test clip, including overlapping clip IDs.
- Added conversations of at least 720 s, 90-s chunks, 90-s detached Transformer-XL
  memory, and checkpointed audio/KV state. The Slurm configuration requests one
  a6 node with four A6000 GPUs for one day.
- GPU training completed InstructS2S and real-video AVQA updates with 4,500
  detached Thinker cache tokens. Checkpoint, optimizer, and XL-state reload
  passed. Fixed development loss fell from 2.104 initially to 0.893 at update 50.
- Added replay-safe activation recomputation, FP32 master weights, homogeneous
  distributed Adam groups, and bit-preserving sharded INT8 expert storage.
  All four GPUs passed frozen-byte and gradient-parity preflights.
- Profiling identified masked attention backward as the main speed limit.
  Full chunks now use the equivalent native lower-right causal kernel; initial
  measured continuation time improved from about 16 s to 7.6 s per update.
  Padded chunks keep the original D2 mask. Dedicated mask and GPU gradient
  parity checks passed.
- Final regression suite: 28 passed, 6 skipped, 4 subtests passed. Job 96800 on
  sls-a6-3 continued past update 50, with a durable update-50 checkpoint and a
  recent median of 7.9 s/update. Padded tail updates remain around 16 s.
- Data and runtime artifacts remain local to D2-Omni. Shared LSKV sources are
  mounted read-only; downloaded raw video provenance is saved alongside data.

## 2026-10-22 — Codex, 10,000-update continuation

- The initial job 96800 completed 1,000 updates successfully. Fixed development
  loss was 0.734 at update 1,000 (2.104 initially); the last 20 updates had a
  median duration of 8.08 s. These losses are development probes, not test QA
  or generated-speech quality measurements.
- Increased the total update limit to 10,000 and submitted job 96803 on one a6
  node with four A6000 GPUs and a one-day limit. It resumes update 1,000 using
  the existing prepared corpus: 63 InstructS2S training conversations and 192
  real-video AVQA training clips.
- Resume permits an increased step limit and changed reporting intervals while
  rejecting training-recipe changes. All eight focused resume tests passed.
  Model weights, optimizer, native D2 learning-rate schedule, data cursors, RNG,
  and detached XL state are restored from the original checkpoint.
- Validation is every 100 updates and checkpointing every 250 to reduce overhead.
  D2's native scheduler is preserved, including its minimum rates after 2,000
  updates. The run also saves at its final update or a handled stop signal.

## 2026-10-22 — Codex, training stopped for inference

- Stopped continuation job 96803 at the user's request. It had not advanced past
  loading, so the latest durable checkpoint remains update 1,000 from job 96800.
- Added inference restoration of the four FSDP adapter shards, retaining the
  frozen waveform renderer and native INT8 streaming expert implementation.
  Focused checkpoint assembly, visual layout, and precision checks: nine passed.
- Submitted inference job 96806 on one A6000 GPU on sls-a6-2. It runs two held-out
  real-video AVQA episodes and the first turn of one held-out InstructS2S
  conversation. Each task's prompt IDs are checked against its training payload.
- Examples supply environment audio and aligned visual features. Text, speech,
  response controls, and self-audio feedback are generated without teacher
  forcing. The AVQA development clips are checked against the training clip IDs.
- Inference job 96806 completed successfully. AVQA generated `C. hail` correctly
  and `C. street` where the reference was `A. On the road`: one exact match out
  of two development examples. Both examples used real video and their original
  training prompts. This is a qualitative check, not an official test score.
- InstructS2S generated a relevant full list of health tips. Its speech lasted
  only 1.92 s and independent recognition captured just the introduction. The
  Talker closed at frame 222 without a new text INTERRUPT. This speech cutoff
  remains an open quality issue; a falling combined loss does not resolve it.
- The full model and waveform renderer occupied 36.25 GiB before the rollouts.
  Eager inference took 4.5–4.7 times the audio duration on one A6000. Saved
  complete timelines, short speech clips, exact prompts, ASR checks, per-frame
  traces, and an HTML review under `runs/inference-step1000/`.

## 2026-10-22 — Codex, speech EOS and generation-stop audit

- Checked all 75 prepared conversations: 3,593 codec BOS/EOS pairs, with every
  EOS exactly one frame after the last supervised speech frame. No open tails.
- Compared all 320 independent AVQA episodes against their read-only source
  metadata with the requested +6-frame retiming; BOS and EOS indices matched.
- EOS is included in CB0 supervision with weight 20 and excluded from residual
  codebook supervision. An actual CE gradient check measured a 20:1 ratio.
  The loss and Talker-span helper functions match native D2's AST exactly.
- Confirmed the InstructS2S inference loop did not stop at text completion:
  last text at 17.44 s, text-finished flag at 17.52 s, Talker closing via its
  codec-EOS path at 17.76 s, and processing continuing to 35.92 s. The reference
  first reply has its final text at 18.48 s and speech EOS at 24.88 s.
- Saved the reproducible audit in `scripts/audit_speech_eos.py` and results in
  `runs/d2-omni-xl-80ms/speech_eos_audit.json`. No training was restarted.

## 2026-10-22 — Codex, 5× speech EOS continuation

- Reduced only the speech-code EOS loss weight from 20 to 5 at the user's request.
  Text RESPONSE/INTERRUPT weights remain 20. Target placement and the native
  Talker trainable inventory are unchanged.
- Added an explicit config override, bounded resume compatibility for this loss
  change, and EOS-weight labels on training/development metrics. Development
  metrics now also report InstructS2S and AVQA separately.
- Preserved the original step-1,000 manifest as
  `runs/d2-omni-xl-80ms/checkpoint-eos20-step1000.json`; its four original shards
  remain intact. Continuation restores weights, optimizer, scheduler, RNG, source
  cursors, and detached XL state.
- All 19 focused checks passed, including the actual 5:1 EOS/ordinary-CB0
  gradient ratio, EOS masking, preserved text control weights, resume guards,
  and distributed Adam parity.
- Submitted job 96812 on sls-a6-2 with four A6000s, a one-day limit, and a 10,000
  total-update target. Configuration and provenance are in `launch-96812.json`.
- Verified restoration at update 1,000 and successful continued updates on both
  AVQA and InstructS2S. Full-chunk update times settled near eight seconds;
  Thinker memory remains detached and bounded at 4,500 tokens.

## 2026-10-22 — Codex, repository cleanup and working-version snapshot

- Removed obsolete AVQA download scratch files and Python/pytest caches;
  preserved download validation receipts under `data/raw/AVQA/` and archived
  the earlier placeholder-only smoke-run logs under `runs/archive/`.
- Preserved datasets, prepared features, training checkpoints, and inference
  outputs outside Git. Shared LSKV sources remain read-only.
- Formatted the new Python modules without changing their ASTs, removed two
  unused imports, documented data/test dependencies, and corrected stale setup,
  source provenance, architecture, and current-status notes.
- Full project validation: 37 tests and four subtests passed; Ruff and shell
  syntax checks passed. Early speech-code EOS during InstructS2S inference
  remains an open issue pending training and another inference check.

## Template

```text
YYYY-MM-DD — name
Completed:
Evidence / artifacts:
Open issues:
Next:
```
