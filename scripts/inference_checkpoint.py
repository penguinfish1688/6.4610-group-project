"""Restore training's four rank-local adapter shards into one inference model."""

import hashlib
import json
from pathlib import Path

import torch


def merge_parameter(parts, shape):
    """FSDP2 shards dimension zero; parameters outside decoder blocks replicate."""
    shape = tuple(shape)
    if all(tuple(part.shape) == shape for part in parts):
        if any(not torch.equal(parts[0], part) for part in parts[1:]):
            raise ValueError("Replicated checkpoint parameters disagree")
        return parts[0]
    if not shape or any(
        part.ndim != len(shape) or tuple(part.shape[1:]) != shape[1:] for part in parts
    ):
        raise ValueError("Checkpoint shard dimensions differ from the model")
    result = torch.cat(parts, dim=0)
    if tuple(result.shape) != shape:
        raise ValueError(f"Merged checkpoint shape {tuple(result.shape)} != {shape}")
    return result


@torch.no_grad()
def restore_inference_checkpoint(model, output):
    output = Path(output)
    manifest = json.loads((output / "latest.json").read_text())
    shards = []
    for entry in manifest["shards"]:
        path = output / entry["path"]
        if path.stat().st_size != entry["bytes"]:
            raise ValueError(f"Checkpoint size mismatch: {path}")
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != entry["sha256"]:
                raise ValueError(f"Checkpoint checksum mismatch: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        if payload["step"] != manifest["step"]:
            raise ValueError("Checkpoint shards have different update counters")
        shards.append(payload)
    named = {name: p for name, p in model.named_parameters() if p.requires_grad}
    if any(set(shard["weights"]) != set(named) for shard in shards):
        raise ValueError("Inference and checkpoint trainable inventories differ")
    for name, parameter in named.items():
        restored = merge_parameter([shard["weights"][name] for shard in shards], parameter.shape)
        if not bool(torch.isfinite(restored).all()):
            raise ValueError(f"Nonfinite checkpoint parameter: {name}")
        parameter.copy_(restored.to(dtype=parameter.dtype, device=parameter.device))
    return dict(step=manifest["step"], manifest=manifest, parameters_restored=len(named))
