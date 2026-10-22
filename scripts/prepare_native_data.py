"""Read shared D2 audio data and write D2-Omni-owned prepared samples."""

from __future__ import annotations

import gc
import hashlib
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch
from types import SimpleNamespace

import torch
import torch.distributed as dist

from d2.hub import asset
from d2.prompts import task_system_prompt_token_ids
from d2_qwen.model.mel import causal_log_mel
from d2_qwen.model.utils import ensure_duplex_control_token
from model.vision_stream import encode_images, placeholder_image
from model.vision_encoder import Qwen3OmniMoeVisionEncoder
from scripts.export_vision_encoder import export_vision_encoder


def shared_audio_loaders(cfg, processor, device, kinds, *, initialize_online=True):
    """Use native D2 materializers with their original control/timing settings."""
    root = Path(cfg["legacy_d2_root"])
    sys.path.insert(0, str(root))
    sys.path.append(cfg["shared_lskv_root"])
    from qwen3.finetune_dataloader import _reference_types
    from lskv.data import build_source_loader, dataset_selection

    native = json.loads((root / "qwen3/config/shared.json").read_text())
    sources = [s for s in native["data"]["datasets"] if s["kind"] in kinds]
    for source in sources:
        for key in ("root", "source_root", "processed_root"):
            if key in source:
                source[key] = str((root / source[key]).resolve())
        source["cache_source_rows"] = source["cache_source_shards"] = 1
        if source["kind"] == "instructs2s_dynamic":
            source["normal_audio_start_offset_frames"] = cfg["control_offset_frames"]
            source["interrupt_audio_end_offset_frames"] = cfg["control_offset_frames"]
        elif source["kind"] == "avqa":
            # The shared SLS V3 copy omitted its small speaker sidecars. Build
            # our own metadata view with byte-identical sidecars from V1.
            shared = Path(source["processed_root"])
            view_root = Path(cfg["prepared_root"]).resolve() / "shared-avqa-metadata"
            view = view_root / "sft_async_v3"
            (view / "speakers").mkdir(parents=True, exist_ok=True)
            manifest = json.loads((shared / "manifest.json").read_text())
            links = {view / name: shared / name for name in ("audio", "rows")}
            copies = {view / name: shared / name for name in ("manifest.json", "pipeline.json")}
            for name, field in (
                ("index.json", "speaker_index_sha256"),
                ("bank.json", "speaker_bank_sha256"),
            ):
                candidate = shared.parent / "sft_async/speakers" / name
                if hashlib.sha256(candidate.read_bytes()).hexdigest() != manifest[field]:
                    raise ValueError(
                        f"Shared speaker sidecar does not match the V3 manifest: {name}"
                    )
                copies[view / "speakers" / name] = candidate
            if not dist.is_initialized() or dist.get_rank() == 0:
                for target, original in copies.items():
                    data = original.read_bytes()
                    if target.is_symlink() or not target.exists() or target.read_bytes() != data:
                        temporary = target.with_name(target.name + f".{os.getpid()}.partial")
                        temporary.write_bytes(data)
                        temporary.replace(target)
            if dist.is_initialized():
                dist.barrier()
            for target, original in links.items():
                try:
                    target.symlink_to(original)
                except FileExistsError:
                    if target.resolve() != original.resolve():
                        raise ValueError(f"Shared-data view points at a different source: {target}")
            source["processed_root"] = str(view)
            source["root"] = str(view_root)
    native["data"]["datasets"] = sources
    native["codec"]["device"] = str(device)
    build, _ = _reference_types()
    from finetune.annotation import configure_online_annotations

    loaders, selections = {}, {}
    for source in sources:
        kind = source["kind"]
        loader_cfg = dict(
            cfg,
            allow_partial_source=(
                kind == "instructs2s_dynamic" and cfg.get("allow_partial_source", False)
            ),
        )
        if kind == "avqa":
            from finetune_async.preprocess.avqa import dataloader as avqa_native

            def owned_manifest(*_args, **_kwargs):
                data = json.loads((Path(source["processed_root"]) / "manifest.json").read_text())
                if data["contract"] != "avqa_async_sft_manifest_v3_unified" or not data["complete"]:
                    raise ValueError("Shared AVQA inventory is not a complete V3 manifest")
                if len(data["rows"]) != data["row_count"]:
                    raise ValueError("Shared AVQA row inventory changed")
                data["_path"] = str(Path(source["processed_root"]) / "manifest.json")
                return data

            # This is our own read-only data view, not a republished V3 build.
            # Native per-row identities and all three audio hashes are still
            # verified by AVQALoader._row/_source_audio/_decode_audio.
            with patch.object(avqa_native, "load_training_manifest", owned_manifest):
                loader = build_source_loader(build, native, source, processor.tokenizer, loader_cfg)
        else:
            loader = build_source_loader(build, native, source, processor.tokenizer, loader_cfg)
        if kind == "avqa":
            from lskv.native_timing import configure_avqa_timing

            configure_avqa_timing(loader, cfg["control_offset_frames"])
        if kind == "instructs2s_dynamic":
            train, probes, _, report = dataset_selection(loader, cfg["world_size"], cfg, kind)
            selections[kind] = dict(
                train=train[: cfg["training_samples"]], probe=probes, report=report
            )
        else:
            # AVQA is selected by official annotation IDs in prepare_avqa below.
            selections[kind] = dict(train=list(range(len(loader))), probe=[])
        loaders[kind] = loader
    reports = (
        configure_online_annotations(
            SimpleNamespace(datasets=tuple(loaders.values())), native, device=device
        )
        if initialize_online
        else []
    )
    return loaders, selections, native, reports


def audio_tensors(example, processor, max_frames):
    if not example.self_audio_playback_delayed:
        raise ValueError("Shared sample is missing D2's self-playback delay")
    ticks = (
        len(example.text_target)
        if max_frames is None
        else min(len(example.text_target), max_frames)
    )
    if not bool((example.text_target[:ticks] == 151669).any()):
        raise ValueError("Prepared prefix contains no assistant RESPONSE")
    batch = {}
    for key, waveform in (("env_mel", example.env_wav_16k), ("self_mel", example.self_wav_16k)):
        mel = causal_log_mel(waveform[: ticks * 1280].cpu().float(), processor.feature_extractor)
        if mel.shape[-1] != ticks * 8:
            raise ValueError("Mel/audio clocks differ")
        batch[key] = mel[None].contiguous()
    for key in ("text_target", "codec_target", "frame_mask", "assistant_mask"):
        batch[key] = getattr(example, key)[:ticks][None].cpu().contiguous()
    if example.interrupt_end_weight_class is not None:
        batch["interrupt_end_weight_class"] = example.interrupt_end_weight_class[:ticks][None].cpu()
    if example.speaker_id is not None:
        batch["speaker_ids"] = torch.tensor([example.speaker_id])
    return batch, ticks


def prepare(cfg, device, log):
    from transformers import Qwen3OmniMoeProcessor

    rank, world = dist.get_rank(), dist.get_world_size()
    release = json.loads((Path(cfg["d2_source"]) / "d2.json").read_text())
    if release["family"] != "qwen" or release["latency_ms"] != 80:
        raise ValueError("D2-Omni must start from the Qwen D2 80-ms checkpoint")
    base = asset(release, "base_model", offline=True)
    processor = Qwen3OmniMoeProcessor.from_pretrained(base, local_files_only=True)
    base_cfg = json.loads((base / "config.json").read_text())
    ensure_duplex_control_token(
        processor.tokenizer, vocab_size=base_cfg["thinker_config"]["text_config"]["vocab_size"]
    )
    prompts = task_system_prompt_token_ids(processor.tokenizer)
    output = Path(cfg["prepared_root"])
    output.mkdir(parents=True, exist_ok=True)
    vision_path = Path("checkpoints/vision_encoder")
    if rank == 0 and not (vision_path / "model.safetensors").is_file():
        export_vision_encoder(base, vision_path)
    dist.barrier()
    encoder = (
        Qwen3OmniMoeVisionEncoder.from_pretrained(
            vision_path, dtype=torch.bfloat16, attn_implementation="sdpa"
        )
        .to(device)
        .eval()
    )
    image = placeholder_image(cfg["vision"]["placeholder_seed"])
    placeholder, deepstack = encode_images(encoder, processor.image_processor, [image])
    log("placeholder_encoded", shape=list(placeholder.shape), deepstack=len(deepstack))
    loaders, selections, native, annotation = shared_audio_loaders(
        cfg, processor, device, {"instructs2s_dynamic"}
    )
    if rank == 0:
        (output / "audio_recipe.json").write_text(
            json.dumps(
                dict(
                    native_config=native,
                    selections=selections,
                    annotation=annotation,
                    prompts={k: list(v) for k, v in prompts.items()},
                    shared_sources_read_only=True,
                ),
                indent=2,
            )
            + "\n"
        )
    rows = []
    exclusions = []
    from finetune.dataloader import InvalidPreparedRowError

    for split in ("probe", "train"):
        for position, index in enumerate(selections["instructs2s_dynamic"][split]):
            if position % world != rank:
                continue
            target = output / f"instructs2s-{split}-{index:06d}.pt"
            if not target.exists():
                log("materializing_audio", split=split, row=index)
                try:
                    example = loaders["instructs2s_dynamic"].exact_active_example(index)
                except InvalidPreparedRowError as error:
                    exclusions.append(dict(split=split, row=index, reason=str(error)))
                    log("source_row_excluded", split=split, row=index, reason=str(error))
                    continue
                batch, ticks = audio_tensors(example, processor, None)
                if ticks * 0.08 < cfg["minimum_conversation_seconds"]:
                    raise ValueError(
                        "Native concatenated InstructS2S conversation is shorter than 720 seconds"
                    )
                batch["system_prompt_input_ids"] = torch.tensor(prompts["conversation"])[None]
                payload = dict(
                    tensors=batch,
                    source="instructs2s_dynamic",
                    task="conversation",
                    source_id=example.sample_id,
                    split=split,
                    ticks=ticks,
                    format="d2_omni.prepared.xl.v1",
                    vision_bank=dict(features=placeholder, deepstack=deepstack, repeat=True),
                    placeholder_seed=cfg["vision"]["placeholder_seed"],
                )
                if split == "probe":
                    payload["environment_audio_16k"] = example.env_wav_16k[: ticks * 1280].cpu()
                temporary = target.with_suffix(".partial")
                torch.save(payload, temporary)
                temporary.replace(target)
                log("sample_prepared", split=split, row=index, ticks=ticks)
            rows.append(
                dict(
                    path=str(target),
                    split=split,
                    source="instructs2s_dynamic",
                    ticks=torch.load(target, mmap=True, weights_only=False)["ticks"],
                )
            )
    (output / f"exclusions-rank{rank}.json").write_text(json.dumps(exclusions, indent=2) + "\n")
    all_rows = [None] * world
    dist.all_gather_object(all_rows, rows)
    rows = [row for rank_rows in all_rows for row in rank_rows]
    rows.sort(key=lambda row: row["path"])
    del loaders
    gc.collect()
    torch.cuda.empty_cache()
    # Keep the frozen visual encoder available if AVQA arrives during the wait.
    return rows, processor, encoder
