# Architecture and integration notes

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

Restoring the tower is necessary but insufficient: the current duplex runtime
does not feed video features into the Thinker.

## Proposed data flow

```text
camera -> timestamped frames -> native/streaming vision encoder -> compression --+
microphone ------------------> D2 causal audio encoder ------------------------+-> Thinker
delayed assistant audio -----> D2 self-audio lane ------------------------------+     |
                                                                                 text
                                                                                   |
                                                                            Talker -> speech
```

Keep the native encoder as a reference until profiling shows whether distillation
is necessary. The extracted tower exposes both merged visual tokens and DeepStack
features. The integration design must account for both, including position IDs,
visual-token masks, and where intermediate features enter the Thinker.

The initial vision budget is one token per 80-ms tick, but this trades spatial
detail against sequence length and latency. Evaluate larger budgets as a baseline.
Camera sampling, encoder execution cadence, and model interaction cadence are
separate settings. Define explicit missing-frame and stale-frame behavior.

The native patch embed groups frames temporally. A group is available only when
its last frame has arrived. Frame-by-frame image processing or a causal temporal
frontend may be useful, but either needs a measured baseline. Training may use a
teacher with richer context; student inputs and evaluation must stay causal.

## Stage II integration checklist

1. Keep or attach the local visual tower during backbone loading.
2. Define the compressed feature shapes and any DeepStack projection layers.
3. Extend data preparation and collation with timestamps, vision features, masks,
   and position information. Preserve D2's existing self-audio delay semantics.
4. Insert visual features in the same causal order during training and inference.
5. Update cache accounting, CUDA graph shapes, and any fixed prefill assumptions.
6. Overfit one video-QA example, then confirm that changing/removing its video
   affects the answer before scaling the dataset.
7. Recheck audio-only quality and interruption behavior after multimodal SFT.

Do not assume the proposal's `2k + 1 -> 3k + 1` prefill estimate describes the
current implementation exactly. The imported runtime documents environment,
delayed-self, and previous-text blocks. Derive the new token/caching budget from
the actual chosen insertion scheme, including visual merging and DeepStack.

The JSON files in `configs/` record starting research choices. There are no
Stage I/Stage II audiovisual training commands yet.
