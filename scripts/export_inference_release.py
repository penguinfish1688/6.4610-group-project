"""Export the exact retained checkpoint and three demo inputs to safe tensors."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from scripts.inference_checkpoint import merge_parameter


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def export_weights(checkpoint, bundle):
    manifest = json.loads((checkpoint / "latest.json").read_text())
    with safe_open(str(bundle / "base_d2/d2.safetensors"), framework="pt", device="cpu") as base:
        shapes = {name: tuple(base.get_slice(name).get_shape()) for name in base.keys()}
    hidden = shapes["self_silence_embed"][0]
    shapes.update(vision_stream_embed=(hidden,), vision_pad_embed=(hidden,))
    payloads = []
    for entry in manifest["shards"]:
        path = checkpoint / entry["path"]
        if path.stat().st_size != entry["bytes"] or digest(path) != entry["sha256"]:
            raise ValueError(f"Source checkpoint checksum mismatch: {path}")
        payload = torch.load(path, weights_only=False, map_location="cpu", mmap=True)
        if payload["step"] != manifest["step"] or set(payload["weights"]) != set(shapes):
            raise ValueError("Source checkpoint step or parameter inventory mismatch")
        payloads.append(payload)
        print(json.dumps(dict(event="verified_checkpoint", file=str(path))), flush=True)
    weights = {}
    for name, shape in shapes.items():
        tensor = merge_parameter([p["weights"][name] for p in payloads], shape)
        if tensor.dtype != torch.float32 or not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"Expected finite FP32 fine-tuned weights: {name}")
        weights[name] = tensor.contiguous()
    output = bundle / "finetuned"
    output.mkdir(exist_ok=True)
    temporary = output / "model.safetensors.tmp"
    save_file(weights, str(temporary), metadata={"format": "pt", "step": str(manifest["step"])})
    os.replace(temporary, output / "model.safetensors")
    path = output / "model.safetensors"
    receipt = dict(
        format="d2_omni.inference.v1", step=manifest["step"], parameters=len(weights),
        numel=sum(math.prod(shape) for shape in shapes.values()), dtype="float32",
        weights=dict(path=path.name, bytes=path.stat().st_size, sha256=digest(path)),
        source_checkpoint=manifest,
    )
    (output / "latest.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(dict(event="weights_exported", **receipt)), flush=True)


def export_examples(config_path, original_run, bundle):
    from transformers import Qwen3OmniMoeProcessor
    from d2_qwen.model.utils import ensure_duplex_control_token
    from scripts.infer_omni import select_cases, write_wav
    from model.vision_stream import placeholder_image

    cfg = json.loads(config_path.read_text())
    processor = Qwen3OmniMoeProcessor.from_pretrained(bundle / "base_model", local_files_only=True)
    base_config = json.loads((bundle / "base_model/config.json").read_text())
    ensure_duplex_control_token(
        processor.tokenizer,
        vocab_size=base_config["thinker_config"]["text_config"]["vocab_size"],
    )
    cases = select_cases(cfg, processor.tokenizer)
    folder = bundle / "examples"
    folder.mkdir(exist_ok=True)
    placeholder_image(cfg["vision"]["placeholder_seed"]).save(folder / "placeholder.png")
    records = []
    for case in cases:
        name = case["name"]
        target = folder / name
        target.mkdir(exist_ok=True)
        before = json.loads((original_run / name / "result.json").read_text())
        write_wav(target / "input.wav", case["audio"], 16000)
        if digest(target / "input.wav") != digest(original_run / name / "input.wav"):
            raise ValueError(f"Current prepared audio differs from the recorded demo: {name}")
        if list(case["prompt_ids"]) != before["system_prompt_input_ids"]:
            raise ValueError("Prepared task prompt differs from the recorded demo")
        if int(case["visual"]["vision_mask"].sum()) != before["visual_tokens"]:
            raise ValueError("Prepared visual timeline differs from the recorded demo")
        tensors = dict(audio=case["audio"].contiguous(), **{
            key: tensor.contiguous() for key, tensor in case["visual"].items()
        })
        save_file(tensors, str(target / "inputs.safetensors"))
        video = None
        if case["video_path"]:
            source = Path(case["video_path"]).resolve()
            video = f"examples/{name}/video.mp4"
            destination = bundle / video
            if not destination.exists():
                os.link(source, destination)
        record = dict(
            name=name, task=case["task"], input=f"examples/{name}/inputs.safetensors",
            input_sha256=digest(target / "inputs.safetensors"),
            original_prepared_source=case["path"], input_frames=case["input_frames"],
            prompt_ids=list(case["prompt_ids"]), prompt=case["prompt"], reference=case["reference"],
            video=video, annotation=case["annotation"],
            reference_response_seconds=case["reference_response_seconds"],
            expected_text=before["generated_text"], visual_tokens=before["visual_tokens"],
        )
        records.append(record)
        reference = bundle / "reference_results" / name
        reference.mkdir(parents=True, exist_ok=True)
        for filename in ("result.json", "output.wav", "response.wav", "output.txt"):
            source = original_run / name / filename
            if source.is_file():
                shutil.copyfile(source, reference / filename)
        print(json.dumps(dict(event="example_exported", name=name, input_audio_matches_demo=True)), flush=True)
    (folder / "manifest.json").write_text(json.dumps(dict(cases=records), indent=2) + "\n")
    public_cfg = {
        key: value for key, value in cfg.items()
        if key not in {"d2_source", "hf_home", "legacy_d2_root", "shared_lskv_root", "prepared_root", "avqa_raw_root", "output"}
    }
    public_cfg["portable_runtime"] = True
    (bundle / "inference-config.json").write_text(json.dumps(public_cfg, indent=2) + "\n")


def main(args):
    torch.set_num_threads(4)
    if args.stage in {"all", "weights"}:
        export_weights(args.checkpoint, args.output)
    if args.stage in {"all", "examples"}:
        export_examples(args.config, args.original_run, args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/d2-omni-xl-80ms"))
    parser.add_argument("--output", type=Path, default=Path("weight/base_1030"))
    parser.add_argument("--config", type=Path, default=Path("runs/inference-step9000/config.json"))
    parser.add_argument("--original-run", type=Path, default=Path("runs/inference-step9000"))
    parser.add_argument("--stage", choices=["all", "weights", "examples"], default="all")
    main(parser.parse_args())
