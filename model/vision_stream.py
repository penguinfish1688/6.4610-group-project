"""Frozen native image encoding and a causal 12-token / 12.5-Hz clock."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image


def visual_schedule(ticks: int, *, start_tick: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Map audio ticks to (one-second image index, token index); -1 is padding.

    Each image is available at its integer-second capture timestamp. The
    25-tick cycle is 12 image tokens, one pad, then 12 image tokens.
    """
    if ticks < 0 or start_tick < 0:
        raise ValueError("tick counts must be nonnegative")
    tick = torch.arange(start_tick, start_tick + ticks, dtype=torch.long)
    image = torch.div(tick * 80, 1000, rounding_mode="floor")
    first_tick = torch.div(image * 1000 + 79, 80, rounding_mode="floor")
    token = tick - first_tick
    return image, torch.where(token < 12, token, -1)


def align_visual_features(
    features, deepstack, ticks: int, *, start_tick: int = 0, repeat: bool = False
):
    """Expand per-image [N,12,H] features onto the native audio timeline."""
    if features.ndim != 3 or features.shape[1] != 12:
        raise ValueError("each sampled image must encode to exactly 12 tokens")
    image, token = visual_schedule(ticks, start_tick=start_tick)
    valid = (token >= 0) & ((image < len(features)) | repeat)
    selected = image.remainder(len(features)) if repeat else image.clamp(max=len(features) - 1)
    result = features[selected, token.clamp(min=0)].clone()
    result[~valid] = 0
    layers = torch.stack([layer[selected, token.clamp(min=0)] for layer in deepstack])
    layers[:, ~valid] = 0
    return dict(
        vision_features=result[None], vision_mask=valid[None], vision_deepstack=layers[:, None]
    )


def placeholder_image(seed: int = 1337) -> Image.Image:
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, size=(96, 128, 3), dtype=np.uint8))


@torch.no_grad()
def encode_images(encoder, image_processor, images, *, batch_size: int = 16):
    features, layers = [], None
    device, dtype = next(encoder.parameters()).device, next(encoder.parameters()).dtype
    for start in range(0, len(images), batch_size):
        resized = [
            image.convert("RGB").resize((128, 96), Image.Resampling.BICUBIC)
            for image in images[start : start + batch_size]
        ]
        inputs = image_processor(images=resized, do_resize=False, return_tensors="pt")
        grid = inputs["image_grid_thw"]
        if not bool(((grid.prod(-1) // encoder.spatial_merge_unit) == 12).all()):
            raise ValueError(f"processor did not preserve the 12-token grid: {grid}")
        output = encoder(
            inputs["pixel_values"].to(device=device, dtype=dtype), grid.to(device=device)
        )
        features.append(output.pooler_output.cpu().reshape(-1, 12, output.pooler_output.shape[-1]))
        if layers is None:
            layers = [[] for _ in output.deepstack_features]
        for dest, value in zip(layers, output.deepstack_features, strict=True):
            dest.append(value.cpu().reshape(-1, 12, value.shape[-1]))
    return torch.cat(features), [torch.cat(layer) for layer in layers]
