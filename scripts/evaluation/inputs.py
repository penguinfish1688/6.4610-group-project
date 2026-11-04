"""Ablate prepared visual inputs without changing audio or the four-slot clock."""

import torch


def ablate_visual(target, condition, donor=None):
    if condition == "correct":
        return {key: value.clone() for key, value in target.items()}
    result = {key: torch.zeros_like(value) for key, value in target.items()}
    if condition == "padding":
        return result
    if condition != "mismatched" or donor is None:
        raise ValueError("Unknown condition or missing mismatched donor")
    if (target["vision_features"].shape[-1] != donor["vision_features"].shape[-1]
            or target["vision_deepstack"].shape[0] != donor["vision_deepstack"].shape[0]):
        raise ValueError("Donor and target visual dimensions differ")
    # Keep timestamps: no stretching, looping, or shifting donor observations.
    # Both clips must supply a real token at this tick; pad unmatched tails.
    ticks = min(target["vision_mask"].shape[1], donor["vision_mask"].shape[1])
    mask = target["vision_mask"][:, :ticks] & donor["vision_mask"][:, :ticks]
    result["vision_mask"][:, :ticks] = mask
    result["vision_features"][:, :ticks] = donor["vision_features"][:, :ticks] * mask[..., None]
    result["vision_deepstack"][:, :, :ticks] = (
        donor["vision_deepstack"][:, :, :ticks] * mask[None, ..., None]
    )
    return result
