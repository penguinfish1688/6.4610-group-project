"""Join real AVQA video to read-only D2 audio with official split checks."""

from __future__ import annotations

import json
import math
from pathlib import Path
import shutil
from types import MethodType
import zipfile

import torch
import torch.distributed as dist

from d2.prompts import task_system_prompt_token_ids
from model.vision_stream import encode_images
from scripts.prepare_native_data import shared_audio_loaders, audio_tensors


def unpack_upload(root):
    root.mkdir(parents=True, exist_ok=True)
    for archive in (
        Path("AVQA.zip"),
        Path("data/raw/AVQA.zip"),
        Path("data/raw/AVQA/AVQA.zip"),
        root / "AVQA.zip",
    ):
        if not archive.is_file() or not zipfile.is_zipfile(archive):
            continue
        marker = root / f"{archive.name}.extracted.json"
        identity = dict(bytes=archive.stat().st_size, mtime_ns=archive.stat().st_mtime_ns)
        if marker.exists() and json.loads(marker.read_text()) == identity:
            continue
        with zipfile.ZipFile(archive) as source:
            for info in source.infolist():
                path = Path(info.filename)
                if info.is_dir() or path.suffix.lower() not in {".mp4", ".webm", ".mkv"}:
                    continue
                target = root / "videos" / path.name
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists() and target.stat().st_size == info.file_size:
                    continue
                temporary = target.with_suffix(target.suffix + ".partial")
                with source.open(info) as incoming, temporary.open("wb") as outgoing:
                    shutil.copyfileobj(incoming, outgoing)
                temporary.replace(target)
        marker.write_text(json.dumps(identity) + "\n")


def ready(root):
    if dist.get_rank() == 0:
        unpack_upload(root)
    dist.barrier()
    return (
        any(root.rglob("*.mp4"))
        and (root / "annotations/train_qa.json").is_file()
        and (root / "annotations/val_qa.json").is_file()
    )


def clip_key(name):
    stem = Path(name).stem
    video, second = stem.rsplit("_", 1)
    return f"{video}_{int(second):06d}"


def sample_images(path, count):
    """Select the latest decoded frame at or before each 1-Hz timestamp."""
    import av

    images, pts = [], []
    previous = None
    previous_time = 0.0
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        origin = float((stream.start_time or 0) * stream.time_base)
        for frame in container.decode(stream):
            timestamp = float(frame.pts * frame.time_base) - origin
            if previous is None:
                previous, previous_time = frame.to_image(), timestamp
            while len(images) < count and len(images) <= timestamp + 1e-6:
                if abs(timestamp - len(images)) < 1e-6:
                    images.append(frame.to_image())
                    pts.append(timestamp)
                else:
                    images.append(previous.copy())
                    pts.append(previous_time)
            previous, previous_time = frame.to_image(), timestamp
            if len(images) == count:
                break
        while len(images) < count and len(images) <= previous_time + 0.15:
            images.append(previous.copy())
            pts.append(previous_time)
    if len(images) != count or any(p > i + 1e-5 for i, p in enumerate(pts)):
        raise ValueError(f"Video cannot supply the complete causal 1-fps timeline: {path}")
    return images, pts


def prepare_avqa(cfg, processor, encoder, device, log):
    root = Path(cfg["avqa_raw_root"])
    official_train = json.loads((root / "annotations/train_qa.json").read_text())
    official_test = json.loads((root / "annotations/val_qa.json").read_text())
    by_clip = {}
    for annotation in official_train:
        by_clip.setdefault(clip_key(annotation["video_name"]), []).append(annotation)
    test_clips = {clip_key(row["video_name"]) for row in official_test}
    videos = {clip_key(p.name): p for p in root.rglob("*.mp4")}
    # The published annotation files share some clip IDs. Reserve every test
    # clip even when it also appears in train, rather than leaking its media.
    overlap = set(by_clip) & test_clips
    log(
        "official_avqa_splits",
        train_qa=len(official_train),
        test_qa=len(official_test),
        train_clips_excluded_for_test_overlap=len(overlap),
    )
    native_cfg = json.loads((Path(cfg["legacy_d2_root"]) / "qwen3/config/shared.json").read_text())
    source = next(row for row in native_cfg["data"]["datasets"] if row["kind"] == "avqa")
    audio_root = (Path(cfg["legacy_d2_root"]) / source["processed_root"]).resolve()
    manifest = json.loads((audio_root / "manifest.json").read_text())
    eligible = []
    target_count = cfg["avqa_training_samples"] + cfg["avqa_probe_samples"]
    for address in manifest["rows"]:
        row_path = audio_root / address["row_path"]
        row = json.loads(row_path.read_text())
        key = clip_key(row["file_name"])
        if key in test_clips or key not in by_clip or key not in videos:
            continue
        matches = [
            a
            for a in by_clip[key]
            if a["question_text"].strip() == row["question"].split("\nChoices:")[0].strip()
            and a["multi_choice"][a["answer"]].strip() == row["answer_choice_text"].strip()
        ]
        if not matches:
            continue
        eligible.append((address["row_index"], key, row, matches[0]))
        if len(eligible) >= target_count:
            break
    if len({row[1] for row in eligible}) < cfg["world_size"] * 2:
        log("avqa_not_ready", matched_rows=len(eligible), available_video_files=len(videos))
        return []
    probe_clips = set(list(dict.fromkeys(row[1] for row in eligible))[: cfg["avqa_probe_samples"]])
    loaders, _, _, _ = shared_audio_loaders(cfg, processor, device, {"avqa"})
    loader = loaders["avqa"]
    original_codec_target = loader._codec_target

    def capture_clean_self(this, waveform, *args, **kwargs):
        this._omni_clean_self_24k = waveform.detach().cpu().clone()
        return original_codec_target(waveform, *args, **kwargs)

    loader._codec_target = MethodType(capture_clean_self, loader)
    prompt = task_system_prompt_token_ids(processor.tokenizer)["avqa"]
    output, rows = Path(cfg["prepared_root"]), []
    for position, (index, key, metadata, annotation) in enumerate(eligible):
        if position % dist.get_world_size() != dist.get_rank():
            continue
        split = "probe" if key in probe_clips else "train"
        target = output / f"avqa-{split}-{index:06d}.pt"
        if not target.exists():
            example = loader.exact_active_example(index)
            batch, ticks = audio_tensors(example, processor, None)
            count = math.ceil(metadata["source_audio"]["resampled_samples_24k"] / 24000)
            images, source_pts = sample_images(videos[key], count)
            features, deepstack = encode_images(encoder, processor.image_processor, images)
            # Native D2 places the scene audio at timeline zero. The following
            # spoken question and answer retain the exact native signal timing.
            end_ms = metadata["source_audio"]["resampled_samples_24k"] * 1000 / 24000
            batch["system_prompt_input_ids"] = torch.tensor(prompt)[None]
            temporary = target.with_suffix(".partial")
            torch.save(
                dict(
                    tensors=batch,
                    task="avqa",
                    source="avqa",
                    split=split,
                    format="d2_omni.prepared.xl.v1",
                    ticks=ticks,
                    vision_bank=dict(
                        features=features, deepstack=deepstack, repeat=False, end_ms=end_ms
                    ),
                    environment_audio_16k=example.env_wav_16k.cpu(),
                    self_audio_16k=example.self_wav_16k.cpu(),
                    assistant_audio_24k=loader._omni_clean_self_24k,
                    official_split="train",
                    official_qa_id=annotation["id"],
                    video_name=key,
                    video_path=str(videos[key]),
                    frame_pts=source_pts,
                ),
                temporary,
            )
            temporary.replace(target)
            log(
                "avqa_sample_prepared",
                row=index,
                split=split,
                video=key,
                ticks=ticks,
                sampled_images=count,
                source_duration_ms=end_ms,
            )
        rows.append(dict(path=str(target), split=split, source="avqa"))
    # Compose complete native QA episodes until each source timeline is >=720 s.
    # Recompute causal mel and continuous Mimi once over the concatenated audio.
    composed = []
    minimum_ticks = math.ceil(cfg["minimum_conversation_seconds"] / 0.08)
    enough = all(
        sum(
            torch.load(row["path"], mmap=True, weights_only=False)["ticks"]
            for row in rows
            if row["split"] == split
        )
        >= minimum_ticks
        for split in ("probe", "train")
    )
    ready_tensor = torch.tensor(int(enough), device=device)
    dist.all_reduce(ready_tensor, op=dist.ReduceOp.MIN)
    if not bool(ready_tensor):
        log("avqa_concatenation_waiting_for_more_clips")
        return []
    for split in ("probe", "train"):
        members = [row for row in rows if row["split"] == split]
        groups, current, ticks = [], [], 0
        for row in members:
            payload = torch.load(row["path"], mmap=True, weights_only=False)
            current.append(row)
            ticks += payload["ticks"]
            if ticks >= minimum_ticks:
                groups.append(current)
                current, ticks = [], 0
        if current:
            if groups:
                groups[-1].extend(current)
            else:
                log(
                    "avqa_concatenation_not_ready",
                    split=split,
                    ticks=ticks,
                    minimum_ticks=minimum_ticks,
                    members=len(members),
                )
                return []
        for group_index, group in enumerate(groups):
            target = output / f"avqa-{split}-xl-r{dist.get_rank()}-g{group_index}.pt"
            if not target.exists():
                payloads = [torch.load(row["path"], mmap=True, weights_only=False) for row in group]
                total = sum(row["ticks"] for row in payloads)
                if total < minimum_ticks:
                    raise ValueError("AVQA concatenation is shorter than 720 seconds")
                from d2_qwen.model.mel import causal_log_mel

                environment = torch.cat([row["environment_audio_16k"] for row in payloads])
                own = torch.cat([row["self_audio_16k"] for row in payloads])
                clean_self = torch.cat([row["assistant_audio_24k"] for row in payloads])
                batch = {}
                for name in (
                    "text_target",
                    "codec_target",
                    "frame_mask",
                    "assistant_mask",
                    "interrupt_end_weight_class",
                ):
                    if name in payloads[0]["tensors"]:
                        batch[name] = torch.cat([row["tensors"][name] for row in payloads], dim=1)
                for name in ("speaker_ids", "system_prompt_input_ids"):
                    batch[name] = payloads[0]["tensors"][name]
                batch["env_mel"] = causal_log_mel(environment.float(), processor.feature_extractor)[
                    None
                ]
                batch["self_mel"] = causal_log_mel(own.float(), processor.feature_extractor)[None]
                # The frozen codec returns inference tensors; materialize a normal
                # tensor before restoring native silence and BOS/EOS positions.
                codes = (
                    loader._online_codec_encoder.encode_aligned_batch([clean_self.float()])[0]
                    .cpu()
                    .long()
                    .clone()
                )
                if codes.shape != (total, 16):
                    raise ValueError("Concatenated Mimi/audio clocks differ")
                codes[~batch["assistant_mask"][0]] = loader.silence_frame
                boundaries = batch["codec_target"][0, :, 0] >= 2048
                codes[boundaries] = batch["codec_target"][0, boundaries]
                batch["codec_target"] = codes[None]
                segments, offset = [], 0
                for row in payloads:
                    bank = row["vision_bank"]
                    segments.append(
                        dict(
                            start_tick=offset,
                            end_tick=offset + row["ticks"],
                            features=bank["features"],
                            deepstack=bank["deepstack"],
                            end_ms=bank["end_ms"],
                        )
                    )
                    offset += row["ticks"]
                temporary = target.with_suffix(".partial")
                torch.save(
                    dict(
                        format="d2_omni.prepared.xl.v1",
                        tensors=batch,
                        ticks=total,
                        vision_segments=segments,
                        source="avqa",
                        split=split,
                        source_members=[row["video_name"] for row in payloads],
                        official_split="train",
                        control_offset_frames=cfg["control_offset_frames"],
                    ),
                    temporary,
                )
                temporary.replace(target)
                log(
                    "avqa_concatenated",
                    split=split,
                    members=len(group),
                    ticks=total,
                    seconds=total * 0.08,
                    path=str(target),
                )
            ticks = torch.load(target, mmap=True, weights_only=False)["ticks"]
            composed.append(dict(path=str(target), split=split, source="avqa", ticks=ticks))
    all_rows = [None] * dist.get_world_size()
    dist.all_gather_object(all_rows, composed)
    return sorted([row for chunk in all_rows for row in chunk], key=lambda row: row["path"])
