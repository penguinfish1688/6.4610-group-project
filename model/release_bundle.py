"""Read a complete, relocatable D2-Omni inference release without shared data."""

import hashlib
import json
from pathlib import Path

from safetensors.torch import load_file

from d2.prompts import task_system_prompt_token_ids


def inside(root, name):
    path = (root / name).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Release path escapes its bundle")
    return path


def inference_config(bundle):
    root = Path(bundle).resolve()
    config = json.loads((root / "inference-config.json").read_text())
    config.update(d2_source=str(root / "base_d2"), output=str(root / "finetuned"), portable_runtime=True)
    return config


def select_cases(bundle, tokenizer, *, manifest_path=None, data_root=None, names=None):
    root = Path(data_root or bundle).resolve()
    manifest = json.loads(Path(manifest_path or root / "examples/manifest.json").read_text())
    prompts = task_system_prompt_token_ids(tokenizer)
    cases = []
    for row in manifest["cases"]:
        if names is not None and row["name"] not in names:
            continue
        path = inside(root, row["input"])
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != row["input_sha256"]:
                raise ValueError(f"Released input checksum mismatch: {path}")
        data = load_file(str(path), device="cpu")
        ids = tuple(row["prompt_ids"])
        if ids != prompts[row["task"]]:
            raise ValueError("Released and training system prompts differ")
        if data["audio"].numel() % 1280:
            raise ValueError("Released audio does not cover complete 80-ms frames")
        frames = data["audio"].numel() // 1280
        visual = {key: data[key] for key in ("vision_features", "vision_mask", "vision_deepstack")}
        if tuple(visual["vision_mask"].shape) != (1, frames):
            raise ValueError("Released audio and visual clocks differ")
        if int(visual["vision_mask"].sum()) != row["visual_tokens"]:
            raise ValueError("Released visual schedule differs from the recorded demo")
        video = inside(root, row["video"]) if row["video"] else None
        if video is not None and not video.is_file():
            raise FileNotFoundError(video)
        cases.append(dict(
            name=row["name"], path=row["original_prepared_source"], task=row["task"],
            audio=data["audio"], visual=visual, input_frames=row["input_frames"],
            prompt_ids=ids, prompt=row["prompt"], reference=row["reference"],
            video_path=str(video) if video else None, annotation=row["annotation"],
            reference_response_seconds=row["reference_response_seconds"],
            split=row.get("split", "development"),
        ))
    return cases
