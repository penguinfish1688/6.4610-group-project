# Architecture and integration notes

## Current first-version contract (CK, 2026-10-22)

- Initialize from the Qwen D2 80-ms release, with its original signal tokens
  and SFT parameter groups. The continuation uses speech-code EOS CE weight 5
  (initial run: 20); text RESPONSE/INTERRUPT weights remain 20. CK explicitly
  selected RMS +6-frame RESPONSE/INTERRUPT placement for both sources.
- Concatenate same-task data to at least 720 seconds. Train with 90-second
  chunks and 90 seconds of detached, frozen KV memory, carrying audio encoder,
  delayed-self, text/control and Talker state across chunks of each conversation.
  Save and restore that recurrence with model/optimizer checkpoints.
- Use only InstructS2S and AVQA. InstructS2S receives a fixed seeded random RGB
  image encoded by the frozen native vision tower. AVQA receives its real video.
- Use `d2.prompts.task_system_prompt_token_ids`: `conversation` for InstructS2S
  and `avqa` for AVQA, retaining the original prompt wording for both tasks.
- Independently encode 1-fps images resized to 96x128 pixels, producing exactly
  12 merged tokens and their matching DeepStack features. No distillation.
- Keep four positions per audio tick: user audio, delayed self audio, vision,
  and previous text. Visual padding occupies a position but has no DeepStack
  injection. Every two seconds contain 24 real visual tokens and one pad.
- Timestamp video features on the same assembled timeline as scene audio.
  Never consume a frame before its capture timestamp; preserve the complete
  source clip rather than substituting a thumbnail or an audio-only AVQA row.
- AVQA `train_qa.json` is training; the official website calls `val_qa.json` the
  test set. Reserve it for final evaluation. Development probes come from
  disjoint source clips/conversations within the training sources.
- All shared LSKV data is read-only. D2-Omni owns its manifests, converted tensors,
  video frames/features, checkpoints, logs, and source symlinks.
- Allocation: one `a6` node, four RTX A6000 GPUs, 24 hours, FSDP sharding.
  The original one-hour AVQA wait fallback is retained in the preparation code;
  the current prepared corpus includes real AVQA video.

Executable work is in `scripts/train_omni.py`, `scripts/prepare_native_data.py`,
`scripts/prepare_avqa.py`, and `scripts/train_omni.sbatch`. Four-GPU training has
completed both source types, retained 4,500 detached Thinker cache tokens, and
passed checkpoint/optimizer/recurrence reload. The initial run completed 1,000
updates; fixed development loss declined from 2.104 to 0.734. Median step time
over its last 20 updates was 8.08 s; padded tail chunks use the slower
explicit-mask path. These are development losses, not official AVQA test results.

Free-running inference uses the exact training prompts and generated self-audio
feedback. At update 1,000, one of two development AVQA examples matched its
reference. InstructS2S generated a coherent full text reply but only spoke its
introduction. The loop continued after text completion; the Talker closed via
speech-code EOS without a new Thinker INTERRUPT. Target placement and native
EOS masking passed a full prepared-corpus audit. The EOS ×5 continuation is an
experiment to address this issue, not a verified fix.

The initial corpus uses 63 InstructS2S training conversations and 192 AVQA
training clips, plus separate development media. It is an initial working
subset of the shared/official sources, not a full-data training run.

Full 80-ms chunks have ordinary causal serialization. The optimized attention
path uses PyTorch's lower-right causal bias to select FlashAttention even with
cached memory; partially padded chunks retain the original explicit mask.
Decoder attention and MLP recomputation preserve frozen KV prefixes and write
each live cache only once. Selected expert payloads use bit-preserving floating
storage for FSDP transport, with INT8 interpretation restored before compute.

## Existing D2

D2's Stage I learns a causal audio encoder by distilling the pretrained encoder.
Stage II trains full-duplex behavior on aligned user/assistant audio timelines.
The Qwen variant operates on native 80-ms frames and retains the pretrained
Thinker/Talker/speech generation components.

Relevant source entry points:

- `model/backbone/qwen3-omni-D2/model/loading.py`: loads Qwen and D2 weights;
  currently deletes `qwen.thinker.visual` to reduce audio-only runtime memory.
- `model/backbone/qwen3-omni-D2/model/duplex.py` and `model/base.py`: model behavior
  and training logic to inspect when adding the visual lane.
- `model/backbone/qwen3-omni-D2/inference/`: streaming inference implementation.
- `model/backbone/qwen3-omni-D2/training/`: prepared-tensor training entry points.
- `model/backbone/d2/SAMPLES.md`: existing audio/control/text tensor contract.

The training wrapper accepts frozen visual features and DeepStack embeddings,
inserting them at the fourth-stream layout's visual positions. The visual tower
is loaded separately during feature preparation.

## Implemented data flow

```text
camera -> timestamped 1-fps images -> resize -> frozen vision encoder -----------+
microphone ------------------> D2 causal audio encoder ------------------------+-> Thinker
delayed assistant audio -----> D2 self-audio lane ------------------------------+     |
                                                                                 text
                                                                                   |
                                                                            Talker -> speech
```

The extracted tower exposes both merged visual tokens and DeepStack features.
The implementation supplies matching visual masks and DeepStack features to
the native Thinker; visual padding uses a learned embedding without DeepStack.

The first-version vision budget is one slot per 80-ms tick, but this trades spatial
detail against sequence length and latency. Evaluate larger budgets as a baseline.
Camera sampling, encoder execution cadence, and model interaction cadence are
separate settings. Define explicit missing-frame and stale-frame behavior.

The native patch embed groups frames temporally. This version encodes each image
independently, duplicating that image across its temporal patch group through the
image processor. It consumes no future video frames. Any future temporal frontend
must preserve this causal availability constraint.

## Remaining evaluation and runtime work

- Verify complete spoken answers after the EOS-weight continuation.
- Measure the effect of changing or removing AVQA video, and evaluate the
  official test split only after selecting the training recipe.
- Recheck audio conversation quality, interruptions, and longer free-running
  audiovisual sessions.
- Adapt native CUDA graphs for DeepStack and measure end-to-end latency.

At the supported 80-ms cadence, each tick has three prefix positions (user audio,
delayed self audio, vision) and one previous-text position. Thinker XL memory
therefore retains 4,500 tokens for 1,125 ticks, in addition to prompt accounting.

`configs/train-sls.json` and `scripts/train_omni.sbatch` define the experimental
SLS training run. Eager visual inference is implemented; the original D2 CUDA
graphs have not been adapted for visual DeepStack injection.
