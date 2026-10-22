"""Free-running D2-Omni speech/video inference on held-out prepared examples."""

from __future__ import annotations

import argparse
import gc
import html
import json
import math
import os
from pathlib import Path
import time
import wave

import numpy as np
import torch
import torch.nn.functional as F

from d2.prompts import task_system_prompt_token_ids
from d2_qwen.inference.session import _MacroSession, SamplingConfig
from d2_qwen.model.loading import load_model
from model.vision_stream import align_visual_features
from scripts.inference_checkpoint import restore_inference_checkpoint
from scripts.quantize_experts import quantize


def write_wav(path, samples, rate):
    data = (samples.detach().cpu().float().clamp(-1, 1).numpy() * 32767).round().astype("<i2")
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes(data.tobytes())


def write_pcm(path, pcm, rate=24000):
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes(pcm)


def select_cases(cfg, tokenizer):
    manifest = json.loads((Path(cfg["output"]) / "data_manifest.json").read_text())
    prepared = Path(cfg["prepared_root"])
    instruct = next(
        row
        for row in manifest["rows"]
        if row["source"] == "instructs2s_dynamic" and row["split"] == "probe"
    )
    video_paths = sorted(
        path for path in prepared.glob("avqa-probe-*.pt") if "-xl-" not in path.name
    )[:2]
    training_videos = set()
    for row in manifest["rows"]:
        if row["source"] == "avqa" and row["split"] == "train":
            data = torch.load(row["path"], mmap=True, weights_only=False)
            training_videos.update(data["source_members"])
    annotations = {
        row["id"]: row
        for row in json.loads(
            (Path(cfg["avqa_raw_root"]) / "annotations/train_qa.json").read_text()
        )
    }
    prompts = task_system_prompt_token_ids(tokenizer)
    paths = [
        ("avqa-1", video_paths[0]),
        ("instructs2s-1", Path(instruct["path"])),
        ("avqa-2", video_paths[1]),
    ]
    cases = []
    for name, path in paths:
        payload = torch.load(path, mmap=True, weights_only=False)
        task = payload["task"]
        saved_prompt = tuple(payload["tensors"]["system_prompt_input_ids"][0].tolist())
        if saved_prompt != prompts[task]:
            raise ValueError(f"Training and inference system prompts differ: {path}")
        if payload["split"] != "probe":
            raise ValueError("Inference examples must be held out from training")
        targets = payload["tensors"]["text_target"][0]
        starts = (targets == 151669).nonzero().flatten().tolist()
        if not starts:
            raise ValueError(f"Example has no reference response: {path}")
        response_start = starts[0]
        ends = (targets == 151670).nonzero().flatten().tolist()
        response_end = next((i for i in ends if i > response_start), len(targets))
        stop = response_start if task == "conversation" else payload["ticks"]
        # Only environment audio is supplied. The model generates its own text,
        # response controls, codec history, waveform, and self-audio feedback.
        audio = payload["environment_audio_16k"][: stop * 1280].float()
        audio = F.pad(audio, (0, stop * 1280 - audio.numel()))
        tail_frames = math.ceil((20 if task == "conversation" else 12) / 0.08)
        audio = F.pad(audio, (0, tail_frames * 1280))
        bank = payload["vision_bank"]
        visual = align_visual_features(
            bank["features"], bank["deepstack"], stop + tail_frames, repeat=bank["repeat"]
        )
        if task == "avqa":
            if (
                payload["video_name"] in training_videos
                or not Path(payload["video_path"]).is_file()
            ):
                raise ValueError("AVQA inference media is missing or belongs to training")
            visual["vision_mask"] &= torch.arange(stop + tail_frames)[None] * 80 < bank["end_ms"]
        excluded = set(tokenizer.all_special_ids) | {151669, 151670}
        reference = tokenizer.decode(
            [int(i) for i in targets[response_start:response_end] if int(i) not in excluded],
            skip_special_tokens=True,
        ).strip()
        annotation = annotations[payload["official_qa_id"]] if task == "avqa" else None
        cases.append(
            dict(
                name=name,
                path=str(path),
                task=task,
                audio=audio,
                visual=visual,
                input_frames=stop,
                prompt_ids=saved_prompt,
                prompt=tokenizer.decode(saved_prompt, skip_special_tokens=False),
                reference=reference,
                video_path=payload.get("video_path"),
                annotation=annotation,
                reference_response_seconds=response_start * 0.08,
            )
        )
    return cases


@torch.no_grad()
def run(args):
    cfg = json.loads(args.config.read_text())
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.manual_seed(cfg["seed"])

    def log(event, **values):
        record = dict(event=event, time=time.time(), job=os.environ.get("SLURM_JOB_ID"), **values)
        with (output / "progress.jsonl").open("a") as stream:
            stream.write(json.dumps(record, default=str) + "\n")
        print(json.dumps(record, default=str), flush=True)

    log("loading_cpu_backbone", checkpoint_run=cfg["output"])
    model, release = load_model(cfg["d2_source"], device="cpu", offline=True, training=True)
    model.enable_vision()
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    log("quantizing_frozen_experts")
    log("experts_ready", **quantize(model, cfg, device, training=False))
    log("restoring_inference_checkpoint")
    checkpoint = restore_inference_checkpoint(model, cfg["output"])
    model.to(device=device).eval().requires_grad_(False)
    gc.collect()
    torch.cuda.empty_cache()
    log(
        "model_ready",
        update=checkpoint["step"],
        parameters_restored=checkpoint["parameters_restored"],
        allocated_gb=torch.cuda.memory_allocated() / 2**30,
    )
    cases = select_cases(cfg, model.processor.tokenizer)
    results = []
    for case in cases:
        torch.manual_seed(cfg["seed"])
        folder = output / case["name"]
        folder.mkdir(exist_ok=True)
        audio = case["audio"]
        frames = audio.numel() // 1280
        visual = {key: value.to(device) for key, value in case["visual"].items()}
        write_wav(folder / "input.wav", audio, 16000)
        if case["video_path"]:
            link = folder / "video.mp4"
            if not link.exists():
                link.symlink_to(Path(case["video_path"]).resolve())
        session = _MacroSession(
            model=model,
            device=device,
            frames_per_unit=1,
            max_context_frames=frames,
            kv_cache_capacity_frames=frames,
            kv_cache_capacity_effective_ms=frames * 80,
            require_cuda_graphs=False,
            cuda_graph_backend=None,
            sampling=SamplingConfig(),
            checkpoint_metadata=dict(update=checkpoint["step"], run=cfg["output"], release=release),
            logical_frames=frames,
            session_id=case["name"],
            system_prompt_kind=case["task"],
            system_prompt_input_ids=case["prompt_ids"],
        )
        log(
            "case_start",
            name=case["name"],
            frames=frames,
            task=case["task"],
            source=case["path"],
            visual_tokens=int(visual["vision_mask"].sum()),
            prompt_matches_training=True,
        )
        pcm, events = [], []
        started = time.monotonic()
        for frame in range(frames):
            vf = dict(
                vision_features=visual["vision_features"][:, frame : frame + 1],
                vision_mask=visual["vision_mask"][:, frame : frame + 1],
                vision_deepstack=visual["vision_deepstack"][:, :, frame : frame + 1],
            )
            produced, records = session.step_macro(audio[frame * 1280 : (frame + 1) * 1280], **vf)
            pcm.extend(produced)
            events.extend(records)
            if frame % 25 == 24:
                log(
                    "frame",
                    name=case["name"],
                    frame=frame + 1,
                    frames=frames,
                    elapsed_seconds=time.monotonic() - started,
                    generated_text="".join(record.get("text_delta", "") for record in events),
                )
        summary = session.close()
        joined = b"".join(pcm)
        write_pcm(folder / "output.wav", joined)
        text = "".join(record.get("text_delta", "") for record in events).strip()
        (folder / "output.txt").write_text(text + "\n")
        signal = np.frombuffer(joined, dtype="<i2").astype(np.float64) / 32768
        result = dict(
            name=case["name"],
            task=case["task"],
            checkpoint_update=checkpoint["step"],
            prepared_source=case["path"],
            split="development",
            prompt=case["prompt"],
            system_prompt_input_ids=list(case["prompt_ids"]),
            prompt_matches_training=True,
            reference_text=case["reference"],
            generated_text=text,
            annotation=case["annotation"],
            video=case["video_path"],
            real_video=case["task"] == "avqa",
            visual_tokens=int(visual["vision_mask"].sum()),
            audio_seconds=frames * 0.08,
            wall_seconds=time.monotonic() - started,
            teacher_forcing=False,
            output_rms=float(np.sqrt(np.mean(signal**2))),
            output_peak=float(np.max(np.abs(signal))),
            response_starts_seconds=[
                e["seconds"] for e in summary["text_trace"] if e["kind"] == "response"
            ],
            session_summary=summary,
            events=events,
        )
        (folder / "result.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
        results.append(result)
        log(
            "case_complete",
            name=case["name"],
            text=text,
            reference=case["reference"],
            wall_seconds=result["wall_seconds"],
            output_rms=result["output_rms"],
        )
        del session, visual
        gc.collect()
        torch.cuda.empty_cache()
    (output / "results.json").write_text(
        json.dumps(dict(checkpoint=checkpoint, examples=results), indent=2, default=str) + "\n"
    )
    cards = []
    for result in results:
        name = result["name"]
        video = (
            f'<video controls muted width="480" src="{name}/video.mp4"></video>'
            if result["real_video"]
            else "<p>Encoded random placeholder image.</p>"
        )
        cards.append(
            f"<article><h2>{html.escape(name)}</h2>{video}"
            f'<p>Input audio</p><audio controls src="{name}/input.wav"></audio>'
            f'<p>Generated speech</p><audio controls src="{name}/output.wav"></audio>'
            f"<p><b>Generated:</b> {html.escape(result['generated_text'])}</p>"
            f"<p><b>Reference:</b> {html.escape(result['reference_text'])}</p>"
            f"<details><summary>Exact training task prompt</summary><pre>{html.escape(result['prompt'])}</pre>"
            f"</details></article>"
        )
    page = (
        '<!doctype html><html><meta charset="utf-8"><title>D2-Omni inference</title>'
        "<style>body{font:18px system-ui;max-width:900px;margin:32px auto;padding:16px}"
        "article{border-top:1px solid #ccc;padding:20px 0}pre{white-space:pre-wrap}</style>"
        f"<h1>D2-Omni checkpoint {checkpoint['step']}</h1>"
        "<p>Free-running inference on development examples. AVQA uses real video and audio; "
        "InstructS2S uses its encoded placeholder image. Each task uses its exact training prompt.</p>"
        + "".join(cards)
        + "</html>"
    )
    (output / "index.html").write_text(page)
    log(
        "complete",
        examples=len(results),
        update=checkpoint["step"],
        report=str(output / "index.html"),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/train-sls.json"))
    parser.add_argument("--output", type=Path, default=Path("runs/inference-step1000"))
    run(parser.parse_args())
