"""Exercise audio alignment, split isolation and the downstream D2 target contract."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch
from transformers import WhisperFeatureExtractor

from scripts.prepare_audio_dataset import (
    BOS,
    EOS,
    RESPONSE,
    assemble_episode,
    make_payload,
    pack_episodes,
    read_manifest,
)
from scripts.xl_data import chunk_tensors


class Tokenizer:
    all_special_ids = [151669, 151670]

    def encode(self, text, **kwargs):
        return list(range(100, 100 + len(text.split())))


class RecordingCodec:
    def __init__(self):
        self.lengths = []

    def encode(self, waveform):
        self.lengths.append(len(waveform))
        return torch.ones(len(waveform) // 1920, 16, dtype=torch.long)


def audio_row(tmp_path):
    # Stereo 48 kHz checks channel reduction and resampling, with silent tails.
    time = np.arange(48000) / 48000
    tone = (0.2 * np.sin(2 * np.pi * 220 * time)).astype(np.float32)
    audio = np.pad(tone, (3840, 3840))
    sf.write(tmp_path / "user.wav", np.stack([audio, audio], axis=-1), 48000)
    sf.write(tmp_path / "answer.wav", tone, 48000)
    return dict(
        id="one",
        group_id="recording-one",
        split="train",
        turns=[
            dict(user_audio="user.wav", assistant_audio="answer.wav", assistant_text="Hello there")
        ],
    )


def test_resampling_controls_and_complete_conversation_codec(tmp_path):
    row = audio_row(tmp_path)
    episode = assemble_episode(row, tmp_path, Tokenizer())
    event = episode["events"][0]
    assert event["interrupt"] == 6
    assert event["response"] == 13 - 1 + 6  # 1 second rounds to 13 native ticks.
    codec = RecordingCodec()
    processor = SimpleNamespace(
        tokenizer=Tokenizer(),
        feature_extractor=WhisperFeatureExtractor(
            feature_size=128, sampling_rate=16000, hop_length=160, n_fft=400
        ),
    )
    bank = dict(features=torch.ones(1, 12, 4), deepstack=[torch.ones(1, 12, 4)] * 3, repeat=True)
    payload = make_payload([episode, episode], processor, codec, bank, 151700)
    assert codec.lengths == [2 * episode["ticks"] * 1920]
    batch = payload["tensors"]
    assert int((batch["text_target"] == RESPONSE).sum()) == 2
    assert int((batch["codec_target"][:, :, 0] == BOS).sum()) == 2
    assert int((batch["codec_target"][:, :, 0] == EOS).sum()) == 2
    assert "speaker_ids" not in batch  # Use model's speaker token, never codec token zero.
    chunk = chunk_tensors(payload, 0, 1125, 151700)
    assert int(chunk["frame_mask"].sum()) == payload["ticks"]
    assert chunk["codec_target"].shape == (1, 1125, 16)
    assert not chunk["vision_mask"][:, payload["ticks"] :].any()


def test_transcript_capacity_never_truncates(tmp_path):
    row = audio_row(tmp_path)
    row["turns"][0]["assistant_text"] = "word " * 100
    with pytest.raises(ValueError, match="never truncates"):
        assemble_episode(row, tmp_path, Tokenizer())


@pytest.mark.parametrize("same_group", [True, False])
def test_cross_split_recordings_rejected(tmp_path, same_group):
    row = audio_row(tmp_path)
    other = dict(
        row,
        id="two",
        split="probe",
        group_id=row["group_id"] if same_group else "different-recording",
    )
    path = tmp_path / "input.jsonl"
    path.write_text(json.dumps(row) + "\n" + json.dumps(other) + "\n")
    with pytest.raises(ValueError, match="overlaps"):
        read_manifest(path)


def test_packing_never_repeats_or_crosses_splits():
    episodes = [
        dict(id=str(index), ticks=ticks, split=split)
        for index, (ticks, split) in enumerate(
            [(6000, "train"), (4000, "train"), (100, "train"), (9500, "probe")]
        )
    ]
    groups = pack_episodes(episodes, 9000)
    assert [[row["id"] for row in group] for group in groups] == [["0", "1", "2"], ["3"]]
    with pytest.raises(ValueError, match="fewer"):
        pack_episodes([dict(id="short", ticks=100, split="train")], 9000)
