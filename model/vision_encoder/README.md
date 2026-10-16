# Qwen3-Omni vision encoder

`modeling_qwen3_omni.py` contains the actual editable vision tower extracted from
Transformers **5.13.0**, matching D2's dependency pin. This includes patch
embedding, position embeddings, attention, MLPs, spatial merging, and DeepStack
features. Its configuration class and generic model utilities come from the
pinned Transformers installation. Provenance and extraction changes are in
[UPSTREAM.json](UPSTREAM.json); the original [Apache-2.0 license](LICENSE) is retained.

The encoder starts with the native forward computation and parameter names. It
is not yet a streaming student, token compressor, or distillation trainer.
Modify this file for Stage I; keep the installed upstream implementation as the
teacher/reference. Small CPU tests compare forward outputs and gradients with
the upstream tower, including DeepStack outputs and eager/SDPA attention.

## Load pretrained vision weights

Given a local snapshot of `Qwen/Qwen3-Omni-30B-A3B-Instruct`, export only the
`thinker.visual.*` tensors. The D2 release identifies its backbone revision as
`26291f793822fb6be9555850f06dfe95f2d7e695`.

```bash
python scripts/export_vision_encoder.py \
  --checkpoint /path/to/Qwen3-Omni-30B-A3B-Instruct \
  --output checkpoints/vision_encoder
```

Run this from the repository root after installing dependencies. It reads local
safetensors files on CPU without constructing the language or speech model.
Use a new or empty output directory. No pretrained weights are included in Git.

```python
import torch
from model.vision_encoder import Qwen3OmniMoeVisionEncoder

encoder = Qwen3OmniMoeVisionEncoder.from_pretrained(
    "checkpoints/vision_encoder",
    dtype=torch.bfloat16,
    attn_implementation="sdpa",
).to("cuda").eval()

# Use the matching Qwen3-Omni processor to prepare flattened pixel patches
# and image_grid_thw / video_grid_thw. These are not raw RGB image tensors.
with torch.inference_mode():
    features = encoder(
        hidden_states=pixel_values.to("cuda", dtype=torch.bfloat16),
        grid_thw=grid_thw.to("cuda"),
    )
# features.pooler_output: merged tokens in the Thinker's embedding dimension.
# features.deepstack_features: additional features injected at intermediate layers.
```

`pixel_values` and `grid_thw` above are processor outputs supplied by the caller;
this is the encoder call pattern, not a complete video-loading script. The
processor's image/video ordering and temporal grouping must be preserved.

## First experiments

1. Profile preprocessing, encoder latency, tokens per frame, and memory on ORCD.
2. Check incremental image processing before changing attention or distilling.
3. Specify when a temporal patch becomes available: every source frame in it
   must have arrived. Do not timestamp a two-frame patch at its first frame.
4. Compare spatial pooling or learned compression at several token budgets.
5. Decide how to preserve or adapt DeepStack features as well as the final merger.
6. Distill a student only after establishing the native encoder/compression baseline.

Tests use tiny random models and do not establish pretrained accuracy or GPU
latency. Stage I training and integration into D2 are separate follow-up tasks.
