"""Checks for the released mel-only AVQA inference adapter."""

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from scripts.evaluation.prepared import PreparedMelStream, inventory, select_cases
from scripts.evaluation.report import digest


def dataset(tmp_path):
    payload = dict(
        tensors=dict(
            env_mel=torch.arange(128 * 32).reshape(1, 128, 32).float() / 1000,
            text_target=torch.tensor([[151669, 34, 13, 151670]]),
            system_prompt_input_ids=torch.tensor([[1, 2]]),
            self_mel=torch.full((1, 128, 32), float("nan")),
            codec_target=torch.full((1, 4, 16), -999),
        ),
        source_members=["heldout_000000"],
        vision_segments=[dict(start_tick=0, end_tick=4, end_ms=160,
                              features=torch.ones(1, 12, 2),
                              deepstack=[torch.ones(1, 12, 2)] * 3)],
    )
    path = tmp_path / "probe.pt"
    torch.save(payload, path)
    row = dict(path=path.name, source="avqa", split="probe", bytes=path.stat().st_size,
               sha256=digest(path), source_members=payload["source_members"])
    (tmp_path / "manifest.json").write_text(json.dumps(dict(
        format="d2_omni.dataset.v1", official_test_used_for_training=False, rows=[row])))
    (tmp_path / "probe.jsonl").write_text(json.dumps(row) + "\n")
    (tmp_path / "train.jsonl").write_text(json.dumps(dict(
        source="avqa", source_members=["train_000010"])) + "\n")
    return path


def test_extracts_environment_only_and_reference_never_becomes_input(tmp_path):
    dataset(tmp_path)
    _, records = inventory(tmp_path)
    tokenizer = SimpleNamespace(all_special_ids=[], decode=lambda ids, **kw: "C. hail")
    with patch("d2.prompts.task_system_prompt_token_ids", return_value={"avqa": (1, 2)}):
        case = select_cases(tmp_path, records, tokenizer)[0]
    assert case["reference"] == "C. hail"
    assert case["environment_mel"].shape == (128, 154 * 8)
    assert bool(torch.isfinite(case["environment_mel"]).all())
    assert not {"self_mel", "codec_target", "text_target"} & case.keys()
    assert case["visual"]["vision_mask"].sum() == 2
    expected = torch.arange(128 * 32).reshape(128, 32).float() / 1000
    torch.testing.assert_close(case["environment_mel"][:, :32], expected)
    assert float(case["environment_mel"][0, -1]) == pytest.approx(float(expected.max()) - 2)


def test_inventory_excludes_other_timestamps_of_training_video(tmp_path):
    dataset(tmp_path)
    (tmp_path / "train.jsonl").write_text(json.dumps(dict(
        source="avqa", source_members=["heldout_999999"])) + "\n")
    assert inventory(tmp_path)[1] == []


def test_corrupted_probe_is_rejected(tmp_path):
    path = dataset(tmp_path)
    _, records = inventory(tmp_path)
    with path.open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ValueError, match="checksum"):
        select_cases(tmp_path, records, SimpleNamespace(encode=lambda *args, **kw: [1, 2]))


def test_cached_stream_feeds_exact_incremental_mel_not_pcm():
    chunks = []

    def encode(chunk, state):
        chunks.append(chunk.clone())
        return torch.ones(1, 2)

    encoder = SimpleNamespace(dtype=torch.float32, stream_uniform_mel=encode)
    original = SimpleNamespace(model=SimpleNamespace(encoder=encoder), device="cpu",
                               encoder_state=object(), state_summary=lambda: {})
    mel = torch.arange(128 * 16).reshape(128, 16).float()
    stream = PreparedMelStream(original, mel)
    stream.push_audio(torch.full((1280,), float("nan")))
    stream.push_audio(torch.zeros(1280))
    torch.testing.assert_close(torch.cat(chunks, dim=1), mel)
    assert stream.state_summary()["samples_received"] == 2560
    with pytest.raises(ValueError, match="ticks"):
        stream.push_audio(torch.zeros(1280))
