"""Construct visual and scene-audio ablations on the native four-slot clock."""

import math

import torch


def ablate_visual(target, condition, donor=None):
    if condition in {"correct", "scene_audio_muted"}:
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


def ablate_scene_audio(case, condition):
    """Mute scene content while retaining the spoken question and self-feedback.

    Cached mel is already normalized. This is a feature-level ablation, not an
    exact recomputation of the frontend from a muted waveform.
    """
    if condition != "scene_audio_muted":
        return case
    if "environment_mel" not in case or "scene_audio_end_ms" not in case:
        raise ValueError("Scene-audio ablation requires prepared mel and scene timing")
    end_ms = float(case["scene_audio_end_ms"])
    if not math.isfinite(end_ms) or end_ms <= 0:
        raise ValueError("Invalid scene-audio duration")
    mel = case["environment_mel"].clone()
    # 10-ms mel hops; also suppress two hops of STFT history crossing the cut.
    end = math.ceil(end_ms / 10) + 2
    if end >= mel.shape[-1]:
        raise ValueError("Scene interval leaves no spoken-question/response timeline")
    mel[:, :end] = -1.5  # Qwen's normalized zero-energy floor (not numeric zero).
    return dict(case, environment_mel=mel, evaluation=dict(
        case["evaluation"], muted_scene_mel_frames=end,
        audio_condition="scene_muted_question_retained",
    ))
