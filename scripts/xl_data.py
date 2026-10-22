"""Native-clock chunk slicing and detached Transformer-XL checkpoint state."""

import copy

import torch
import torch.nn.functional as F

from d2_qwen.model.constants import CODEC_SILENCE_FRAME
from model.vision_stream import align_visual_features


def chunk_tensors(payload, start, count, pad_id):
    """Keep full source timelines on CPU and transfer only the current chunk."""
    total = payload["ticks"]
    valid = max(0, min(count, total - start))
    end = start + valid
    result = {}
    for name, value in payload["tensors"].items():
        if name == "system_prompt_input_ids":
            if start == 0:
                result[name] = value
        elif name == "speaker_ids":
            result[name] = value
        elif name.endswith("_mel"):
            result[name] = F.pad(value[:, :, start * 8 : end * 8], (0, (count - valid) * 8))
        elif name == "codec_target":
            codes = torch.tensor(CODEC_SILENCE_FRAME).view(1, 1, -1).expand(1, count, -1).clone()
            codes[:, :valid] = value[:, start:end]
            result[name] = codes
        else:
            fill = pad_id if name == "text_target" else 0
            result[name] = F.pad(value[:, start:end], (0, count - valid), value=fill)
    bank = payload.get("vision_bank")
    if bank is not None:
        result.update(
            align_visual_features(
                bank["features"], bank["deepstack"], count, start_tick=start, repeat=bank["repeat"]
            )
        )
        if "end_ms" in bank:
            result["vision_mask"] &= torch.arange(start, start + count)[None] * 80 < bank["end_ms"]
    else:
        segments = payload["vision_segments"]
        hidden = segments[0]["features"].shape[-1]
        layers = len(segments[0]["deepstack"])
        result.update(
            vision_features=torch.zeros(1, count, hidden, dtype=torch.bfloat16),
            vision_mask=torch.zeros(1, count, dtype=torch.bool),
            vision_deepstack=torch.zeros(layers, 1, count, hidden, dtype=torch.bfloat16),
        )
        for segment in segments:
            left = max(start, segment["start_tick"])
            right = min(start + count, segment["end_tick"])
            if left >= right:
                continue
            local = left - segment["start_tick"]
            part = align_visual_features(
                segment["features"], segment["deepstack"], right - left, start_tick=local
            )
            part["vision_mask"] &= (
                torch.arange(local, local + right - left)[None] * 80 < segment["end_ms"]
            )
            a, b = left - start, right - start
            result["vision_features"][:, a:b] = part["vision_features"]
            result["vision_mask"][:, a:b] = part["vision_mask"]
            result["vision_deepstack"][:, :, a:b] = part["vision_deepstack"]
    result["vision_mask"] &= result["frame_mask"]
    return result


def move_state(value, device, memo=None):
    """Copy data-only detached cache/encoder state while preserving aliases."""
    if memo is None:
        memo = {}
    if id(value) in memo:
        return memo[id(value)]
    if torch.is_tensor(value):
        result = value.detach().to(device, copy=True)
    elif isinstance(value, dict):
        result = {key: move_state(item, device, memo) for key, item in value.items()}
    elif isinstance(value, list):
        result = [move_state(item, device, memo) for item in value]
    elif isinstance(value, tuple):
        result = tuple(move_state(item, device, memo) for item in value)
    elif hasattr(value, "__dict__") and not isinstance(value, type):
        result = copy.copy(value)
        memo[id(value)] = result
        for key, item in vars(value).items():
            setattr(result, key, move_state(item, device, memo))
    else:
        result = value
    memo[id(value)] = result
    return result
