"""Build D2-Omni training tensors from paired user/assistant audio and transcripts."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import torch
import torch.nn.functional as F

from d2.prompts import task_system_prompt_token_ids
from d2_qwen.model.constants import CODEC_SILENCE_FRAME
from d2_qwen.model.mel import causal_log_mel
from d2_qwen.model.utils import ensure_duplex_control_token

MIMI_REVISION = "89091b3e466eb6a9d11e537bf26b144f194978f7"
RESPONSE, INTERRUPT, BOS, EOS = 151669, 151670, 2149, 2150
SAMPLES_PER_TICK = 1920


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_audio(path):
    values, rate = sf.read(path, dtype="float32", always_2d=True)
    values = values.mean(axis=1)
    if len(values) == 0 or not np.isfinite(values).all():
        raise ValueError(f"Empty or nonfinite audio: {path}")
    if np.max(np.abs(values)) > 1.0001:
        raise ValueError(f"Audio exceeds [-1,1]; normalize the source: {path}")
    factor = math.gcd(rate, 24000)
    if rate != 24000:
        values = resample_poly(values, 24000 // factor, rate // factor).astype(np.float32)
    # RMS bounds on the native clock; retain complete audible frames.
    padded = np.pad(values, (0, (-len(values)) % SAMPLES_PER_TICK))
    rms = np.sqrt(np.mean(padded.reshape(-1, SAMPLES_PER_TICK).astype(np.float64) ** 2, axis=1))
    active = np.flatnonzero(rms >= 10 ** (-60 / 20))
    if len(active) == 0:
        raise ValueError(f"No audible frame above -60 dBFS: {path}")
    return torch.from_numpy(
        padded[active[0] * SAMPLES_PER_TICK : (active[-1] + 1) * SAMPLES_PER_TICK].copy()
    )


def assemble_episode(row, root, tokenizer):
    """Deterministic non-overlapping dialogue; do not infer roles from a mixed recording."""
    parts, events, cursor, provenance = [], [], 0, []
    for turn in row["turns"]:
        user_path, assistant_path = root / turn["user_audio"], root / turn["assistant_audio"]
        user, assistant = read_audio(user_path), read_audio(assistant_path)
        lexical = tokenizer.encode(turn["assistant_text"], add_special_tokens=False)
        if not lexical or set(lexical) & set(tokenizer.all_special_ids):
            raise ValueError("Assistant text must contain lexical tokens and no special tokens")
        user_ticks, assistant_ticks = len(user) // 1920, len(assistant) // 1920
        if user_ticks < 2:
            raise ValueError("User audio needs at least two audible 80-ms frames")
        # Both controls use +6 from the first/last audible user frame.
        interrupt, response = cursor + 6, cursor + user_ticks - 1 + 6
        speech_start = response + 1
        speech_end = speech_start + assistant_ticks
        if len(lexical) > assistant_ticks:
            raise ValueError(
                f"Transcript has {len(lexical)} tokens but only {assistant_ticks} audio frames; "
                "correct transcript/audio alignment (the pipeline never truncates text)"
            )
        events.append(
            dict(
                interrupt=interrupt,
                response=response,
                speech_start=speech_start,
                speech_end=speech_end,
                lexical=lexical,
            )
        )
        parts.append((cursor, user, speech_start, assistant))
        provenance.append(
            dict(
                user_audio=turn["user_audio"],
                user_sha256=sha256(user_path),
                assistant_audio=turn["assistant_audio"],
                assistant_sha256=sha256(assistant_path),
                assistant_text=turn["assistant_text"],
            )
        )
        cursor = speech_end + 8  # EOS then enough silence for playback delay / next user.
    environment, own = torch.zeros(cursor * 1920), torch.zeros(cursor * 1920)
    for start, user, speech_start, assistant in parts:
        environment[start * 1920 : start * 1920 + len(user)] = user
        own[speech_start * 1920 : speech_start * 1920 + len(assistant)] = assistant
    return dict(
        environment=environment,
        assistant=own,
        ticks=cursor,
        events=events,
        id=row["id"],
        group_id=row["group_id"],
        split=row["split"],
        files=provenance,
    )


def pack_episodes(episodes, minimum_ticks):
    groups = []
    for split in ("train", "probe"):
        current, ticks, split_groups = [], 0, []
        for episode in episodes:
            if episode["split"] != split:
                continue
            current.append(episode)
            ticks += episode["ticks"]
            if ticks >= minimum_ticks:
                split_groups.append(current)
                current, ticks = [], 0
        if current:
            if not split_groups:
                raise ValueError(f"{split}: fewer than {minimum_ticks * 0.08:g} seconds of data")
            split_groups[-1].extend(current)
        groups.extend(split_groups)
    return groups


class MimiEncoder:
    """First 16 native codebooks, continuous convolution and transformer state."""

    def __init__(self, source, revision, device, chunk_frames=65):
        from transformers import MimiModel

        self.device, self.chunk_frames = torch.device(device), chunk_frames
        if chunk_frames < 1:
            raise ValueError("Mimi chunk frames must be positive")
        self.model = MimiModel.from_pretrained(
            source, revision=revision if not Path(source).is_dir() else None
        ).eval()
        cfg = self.model.config
        if (cfg.sampling_rate, cfg.frame_rate, cfg.codebook_size) != (24000, 12.5, 2048):
            raise ValueError("Expected native 24-kHz, 12.5-Hz, 2048-entry Mimi")
        if not cfg.use_causal_conv or cfg.audio_channels != 1 or cfg.num_quantizers < 16:
            raise ValueError("Expected a causal mono Mimi encoder with at least 16 codebooks")
        for name in ("decoder_transformer", "decoder", "upsample"):
            delattr(self.model, name)
        self.model.requires_grad_(False).to(self.device)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = False

    @torch.inference_mode()
    def encode(self, waveform):
        if waveform.ndim != 1 or len(waveform) % 1920:
            raise ValueError("Mimi input must be a mono whole-tick 24-kHz timeline")
        past, padding, codes = None, None, []
        for start in range(0, len(waveform), self.chunk_frames * 1920):
            block = waveform[start : start + self.chunk_frames * 1920].to(self.device)
            result = self.model.encode(
                block[None, None],
                num_quantizers=16,
                encoder_past_key_values=past,
                padding_cache=padding,
                use_streaming=True,
                return_dict=True,
            )
            codes.append(result.audio_codes[0].T.cpu())
            past, padding = result.encoder_past_key_values, result.padding_cache
        result = torch.cat(codes).long()
        if result.shape != (len(waveform) // 1920, 16):
            raise ValueError("Mimi/audio clocks differ")
        if int(result.min()) < 0 or int(result.max()) >= 2048:
            raise ValueError("Mimi returned an out-of-range code")
        return result


def make_payload(group, processor, encoder, vision_bank, pad_id):
    ticks = sum(episode["ticks"] for episode in group)
    environment = torch.cat([episode["environment"] for episode in group])
    clean_self = torch.cat([episode["assistant"] for episode in group])
    # Re-encode the complete assembled timeline; never concatenate independently encoded codes.
    codes = encoder.encode(clean_self).clone()
    text = torch.full((ticks,), pad_id, dtype=torch.long)
    activity = torch.zeros(ticks, dtype=torch.bool)
    boundary_events, offset = [], 0
    for episode in group:
        for event in episode["events"]:
            interrupt, response = offset + event["interrupt"], offset + event["response"]
            start, end = offset + event["speech_start"], offset + event["speech_end"]
            text[interrupt], text[response] = INTERRUPT, RESPONSE
            text[response + 1 : response + 1 + len(event["lexical"])] = torch.tensor(
                event["lexical"]
            )
            activity[start:end] = True
            boundary_events.append((response, end))
        offset += episode["ticks"]
    codes[~activity] = torch.tensor(CODEC_SILENCE_FRAME)
    for response, end in boundary_events:
        codes[response, 0], codes[end, 0] = BOS, EOS
    env16 = torch.from_numpy(resample_poly(environment.numpy(), 2, 3).astype(np.float32))
    self16 = torch.from_numpy(resample_poly(clean_self.numpy(), 2, 3).astype(np.float32))
    # Code2Wav physical playback delay; the model supplies its own extra 80-ms delay.
    self16 = F.pad(self16, (371, 0))[: ticks * 1280]
    batch = dict(
        env_mel=causal_log_mel(env16, processor.feature_extractor)[None],
        self_mel=causal_log_mel(self16, processor.feature_extractor)[None],
        text_target=text[None],
        codec_target=codes[None],
        frame_mask=torch.ones(1, ticks, dtype=torch.bool),
        assistant_mask=activity[None],
        interrupt_end_weight_class=torch.zeros(1, ticks, dtype=torch.long),
        system_prompt_input_ids=torch.tensor(
            task_system_prompt_token_ids(processor.tokenizer)["conversation"]
        )[None],
    )
    validate_targets(batch, pad_id)
    return dict(
        format="d2_omni.prepared.xl.v1",
        tensors=batch,
        ticks=ticks,
        source="custom_audio",
        task="conversation",
        split=group[0]["split"],
        vision_bank=vision_bank,
        placeholder_seed=1337,
        source_members=[episode["id"] for episode in group],
        control_offset_frames=6,
    )


def validate_targets(batch, pad_id):
    from d2_qwen.model.base import validate_async_text_targets, talker_state_mask_with_initial

    validate_async_text_targets(
        batch["text_target"], pad_id=pad_id, assistant_start_id=RESPONSE, assistant_end_id=INTERRUPT
    )
    codec0 = batch["codec_target"][:, :, 0]
    mask, opened = talker_state_mask_with_initial(codec0, codec_bos_id=BOS, codec_eos_id=EOS)
    assert not opened.any()
    assert torch.equal(batch["text_target"] == RESPONSE, codec0 == BOS)
    assert not (batch["assistant_mask"] & (~mask | (codec0 == BOS) | (codec0 == EOS))).any()
    assert batch["env_mel"].shape[-1] == batch["text_target"].shape[-1] * 8
    assert batch["self_mel"].shape == batch["env_mel"].shape


def read_manifest(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    identities, groups, files = set(), {}, {}
    for row in rows:
        if not isinstance(row.get("id"), str) or not row["id"] or row["id"] in identities:
            raise ValueError("Every episode needs a unique nonempty string id")
        identities.add(row["id"])
        if row.get("split") not in {"train", "probe"} or not row.get("group_id"):
            raise ValueError("Each episode needs explicit train/probe split and group_id")
        group_id = row["group_id"]
        if group_id in groups and groups[group_id] != row["split"]:
            raise ValueError("Source group overlaps train/probe splits")
        groups[group_id] = row["split"]
        if not isinstance(row.get("turns"), list) or not row["turns"]:
            raise ValueError("Each episode needs at least one paired turn")
        for turn in row["turns"]:
            if set(turn) != {"user_audio", "assistant_audio", "assistant_text"}:
                raise ValueError("Each turn needs user_audio, assistant_audio, assistant_text only")
            if not all(isinstance(value, str) and value.strip() for value in turn.values()):
                raise ValueError("Turn paths and assistant text must be nonempty strings")
            for name in ("user_audio", "assistant_audio"):
                identity = sha256(path.parent / turn[name])
                if identity in files and files[identity] != row["split"]:
                    raise ValueError("Identical source audio overlaps train/probe splits")
                files[identity] = row["split"]
    if not rows:
        raise ValueError("Empty input manifest")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, default=Path("weight/base_1030"))
    parser.add_argument("--mimi", default="kyutai/mimi")
    parser.add_argument("--mimi-revision", default=MIMI_REVISION)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--minimum-seconds", type=float, default=720)
    parser.add_argument("--codec-chunk-frames", type=int, default=65)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a new output directory; existing datasets are immutable")
    if not math.isfinite(args.minimum_seconds) or args.minimum_seconds <= 0:
        raise ValueError("minimum-seconds must be positive and finite")
    rows = read_manifest(args.input)
    from transformers import Qwen3OmniMoeProcessor
    from model.vision_encoder import Qwen3OmniMoeVisionEncoder
    from model.vision_stream import encode_images, placeholder_image

    processor = Qwen3OmniMoeProcessor.from_pretrained(
        args.bundle / "base_model", local_files_only=True
    )
    cfg = json.loads((args.bundle / "base_model/config.json").read_text())
    pad_id = ensure_duplex_control_token(
        processor.tokenizer, vocab_size=cfg["thinker_config"]["text_config"]["vocab_size"]
    )
    episodes = []
    for row in rows:
        episode = assemble_episode(row, args.input.parent, processor.tokenizer)
        episodes.append(
            {
                key: value
                for key, value in episode.items()
                if key not in {"environment", "assistant"}
            }
        )
    del episode
    groups = pack_episodes(episodes, math.ceil(args.minimum_seconds / 0.08))
    vision = Qwen3OmniMoeVisionEncoder.from_pretrained(args.bundle / "vision_encoder")
    vision.to(device=args.device, dtype=torch.bfloat16).eval()
    features, deepstack = encode_images(vision, processor.image_processor, [placeholder_image()])
    vision_bank = dict(features=features, deepstack=deepstack, repeat=True)
    del vision
    encoder = MimiEncoder(args.mimi, args.mimi_revision, args.device, args.codec_chunk_frames)
    (args.output / "prepared").mkdir(parents=True)
    records = []
    by_id = {row["id"]: row for row in rows}
    for index, group in enumerate(groups):
        materialized = [
            assemble_episode(by_id[row["id"]], args.input.parent, processor.tokenizer)
            for row in group
        ]
        payload = make_payload(materialized, processor, encoder, vision_bank, pad_id)
        del materialized
        path = args.output / "prepared" / f"{payload['split']}-{index:06d}.pt"
        torch.save(payload, path)
        records.append(
            dict(
                path=path.relative_to(args.output).as_posix(),
                source="custom_audio",
                split=payload["split"],
                ticks=payload["ticks"],
                bytes=path.stat().st_size,
                sha256=sha256(path),
            )
        )
        print(f"Prepared {index + 1}/{len(groups)}: {path}", flush=True)
    manifest = dict(
        format="d2_omni.dataset.v1",
        rows=records,
        source_pattern=["custom_audio"],
        official_test_used_for_training=False,
    )
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    counts = Counter(row["split"] for row in records)
    ready = all(counts[split] >= 4 for split in ("train", "probe")) and all(
        row["ticks"] >= 9000 for row in records
    )
    provenance = dict(
        format="d2_omni.paired_audio.v1",
        input_sha256=sha256(args.input),
        mimi=args.mimi,
        mimi_revision=args.mimi_revision,
        codec_chunk_frames=args.codec_chunk_frames,
        training_ready=ready,
        conversation_counts=dict(counts),
        episodes=[
            {
                key: value
                for key, value in episode.items()
                if key not in {"environment", "assistant"}
            }
            for episode in episodes
        ],
    )
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    config = json.loads(Path("configs/train-step9000-phase2.json").read_text())
    config.update(
        prepared_manifest=str((args.output / "manifest.json").resolve()),
        prepared_root=str((args.output / "prepared").resolve()),
        d2_source=str((args.bundle / "base_d2").resolve()),
        source_pattern=["custom_audio"],
        output=str((args.output / "training-run").resolve()),
    )
    (args.output / "train-config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                output=str(args.output),
                training_ready=ready,
                note="Four >=720-second conversations per split required by the trainer.",
            )
        )
    )


if __name__ == "__main__":
    main()
