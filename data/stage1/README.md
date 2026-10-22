# Stage I data — CK

The first version uses the frozen native vision encoder without distillation.
Sample AVQA video at 1 fps and resize each image to 96x128 pixels to obtain
12 merged tokens plus the matching DeepStack features. InstructS2S uses a
fixed seeded random placeholder image encoded by the same tower.

`scripts/prepare_native_data.py` and `scripts/prepare_avqa.py` encode the images
on the allocated GPU and store frozen feature banks. `scripts/xl_data.py` places
them on each chunk's audio timeline while preserving the visual schedule across
Transformer-XL boundaries. Feature caching is equivalent to re-encoding these
fixed images because the visual encoder is frozen and runs in evaluation mode.

Use the [draft clip record](../examples/stage1.jsonl). Record source provenance,
split, preprocessing, and encoder revision next to cached features. GPU training
and latency validation are ongoing.
