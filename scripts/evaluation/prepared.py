"""Adapt released AVQA training tensors to independent free-running QA cases.

Only environment mel is an inference input. Teacher text is decoded solely as
the scoring reference; teacher self_mel/codec targets never enter generation.
"""

import json
from pathlib import Path

from scripts.evaluation.report import digest, parse_choice


def inventory(root):
    root = Path(root).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("format") != "d2_omni.dataset.v1":
        raise ValueError("Unsupported training dataset format")
    if manifest.get("official_test_used_for_training") is not False:
        raise ValueError("Release must exclude official test media")
    catalogs = {}
    for split in ("train", "probe"):
        catalogs[split] = [json.loads(line) for line in (root / f"{split}.jsonl").read_text().splitlines()
                           if line.strip()]
    train_clips = {clip for row in catalogs["train"] if row["source"] == "avqa"
                   for clip in row["source_members"]}
    train_sources = {clip.rsplit("_", 1)[0] for clip in train_clips}
    records = []
    for row in catalogs["probe"]:
        if row["source"] != "avqa":
            continue
        for index, clip in enumerate(row["source_members"]):
            if clip in train_clips:
                raise ValueError(f"Training/probe clip overlap: {clip}")
            # Also exclude another timestamp from a training YouTube video.
            if clip.rsplit("_", 1)[0] in train_sources:
                continue
            records.append(dict(
                name=f"{Path(row['path']).stem}-{index:03d}", task="avqa",
                input=row["path"], video=clip, source_video_id=clip.rsplit("_", 1)[0],
                reference_from_targets=True, segment_index=index,
                clip_id=clip, split="development",
            ))
    return manifest, records


def select_cases(root, records, tokenizer):
    import torch
    from d2.prompts import task_system_prompt_token_ids
    from model.vision_stream import align_visual_features

    root = Path(root).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    metadata = {row["path"]: row for row in manifest["rows"]}
    prompts = task_system_prompt_token_ids(tokenizer)
    loaded = {}
    cases = []
    for record in records:
        name = record["input"]
        meta = metadata[name]
        path = (root / name).resolve()
        if not path.is_relative_to(root) or meta["source"] != "avqa" or meta["split"] != "probe":
            raise ValueError("Expected a contained, held-out AVQA tensor file")
        if name not in loaded:
            if path.stat().st_size != meta["bytes"] or digest(path) != meta["sha256"]:
                raise ValueError(f"Dataset checksum mismatch: {path}")
            loaded[name] = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
        payload = loaded[name]
        segment = payload["vision_segments"][record["segment_index"]]
        if payload["source_members"][record["segment_index"]] != record["clip_id"]:
            raise ValueError("Catalog and tensor segment identities differ")
        left, right = segment["start_tick"], segment["end_tick"]
        ticks = right - left
        targets = payload["tensors"]["text_target"][0, left:right]
        starts = (targets == 151669).nonzero().flatten().tolist()
        ends = (targets == 151670).nonzero().flatten().tolist()
        if len(starts) != 1:
            raise ValueError(f"Expected exactly one reference response: {record['name']}")
        stop = next((i for i in ends if i > starts[0]), len(targets))
        excluded = set(tokenizer.all_special_ids) | {151669, 151670}
        reference = tokenizer.decode([int(i) for i in targets[starts[0]:stop]
                                      if int(i) not in excluded], skip_special_tokens=True).strip()
        if parse_choice(reference) is None:
            raise ValueError(f"Unscorable reference for {record['name']}: {reference!r}")
        prompt_ids = tuple(payload["tensors"]["system_prompt_input_ids"][0].tolist())
        if prompt_ids != prompts["avqa"]:
            raise ValueError("Prepared prompt differs from native AVQA prompt")
        # Same 12-second response tail as the original AVQA rollout script.
        tail_ticks = 150
        full_mel = payload["tensors"]["env_mel"][0]
        # The causal frontend uses (max(log10(power), running_peak - 8) + 4)/4.
        # Preserve its prefix normalization when appending silent input frames.
        floor = max(-1.5, float(full_mel[:, :right * 8].max()) - 2.0)
        silence = torch.full((128, tail_ticks * 8), floor, dtype=full_mel.dtype)
        mel = torch.cat((full_mel[:, left * 8:right * 8], silence), dim=1)
        visual = align_visual_features(segment["features"], segment["deepstack"], ticks + tail_ticks)
        visual["vision_mask"] &= torch.arange(ticks + tail_ticks)[None] * 80 < segment["end_ms"]
        cases.append(dict(
            name=record["name"], task="avqa", path=str(path), split="development",
            # Zero PCM is a tick-count placeholder, never used as environment audio.
            audio=torch.zeros((ticks + tail_ticks) * 1280), environment_mel=mel,
            visual=visual, prompt_ids=prompt_ids,
            prompt=tokenizer.decode(prompt_ids, skip_special_tokens=False),
            reference=reference, annotation=None, video_path=None,
            scene_audio_end_ms=segment["end_ms"],
            reference_response_seconds=starts[0] * 0.08,
        ))
    return cases


class PreparedMelStream:
    """Environment-only adapter; generated self-audio still uses real waveforms."""

    def __init__(self, original, mel, ticks_per_call=1):
        self.original = original
        self.mel = mel
        self.cursor = 0
        self.ticks_per_call = ticks_per_call

    def push_audio(self, samples):
        import torch

        count = self.ticks_per_call * 8
        if samples.numel() != self.ticks_per_call * 1280 or self.cursor + count > self.mel.shape[1]:
            raise ValueError("Prepared mel and inference ticks differ")
        encoder = self.original.model.encoder
        chunk = self.mel[:, self.cursor:self.cursor + count]
        self.cursor += count
        with torch.no_grad():
            return encoder.stream_uniform_mel(
                chunk.to(device=self.original.device, dtype=encoder.dtype),
                self.original.encoder_state,
            )

    def state_summary(self):
        summary = self.original.state_summary()
        summary.update(input_mode="prepared_environment_mel", samples_received=self.cursor * 160,
                       cached_mel_frames_consumed=self.cursor)
        return summary
