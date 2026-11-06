"""Portable dataset integrity and ordering at the trainer boundary."""

import hashlib
import json

import pytest

from scripts.prepared_dataset import load_prepared_manifest


def fixture_manifest(tmp_path):
    rows = []
    for name, split in [("z.pt", "probe"), ("a.pt", "train")]:
        content = name.encode()
        (tmp_path / name).write_bytes(content)
        rows.append(
            dict(
                path=name,
                split=split,
                source="avqa",
                ticks=9000,
                bytes=len(content),
                sha256=hashlib.sha256(content).hexdigest(),
            )
        )
    manifest = dict(
        format="d2_omni.dataset.v1",
        rows=rows,
        source_pattern=["avqa"],
        official_test_used_for_training=False,
    )
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path, manifest


def test_relocated_dataset_preserves_rank_order(tmp_path):
    path, _ = fixture_manifest(tmp_path)
    rows = load_prepared_manifest(path, ["avqa"])
    assert [row["path"] for row in rows] == [str(tmp_path / "z.pt"), str(tmp_path / "a.pt")]
    assert [row["split"] for row in rows] == ["probe", "train"]


def test_same_size_corruption_is_rejected(tmp_path):
    path, _ = fixture_manifest(tmp_path)
    (tmp_path / "a.pt").write_bytes(b"oops")
    with pytest.raises(ValueError, match="checksum"):
        load_prepared_manifest(path, ["avqa"])


@pytest.mark.parametrize(
    "change, message",
    [
        ({"path": "../outside.pt"}, "escapes"),
        ({"ticks": 8999}, "shorter"),
        ({"split": "test"}, "source or split"),
    ],
)
def test_invalid_rows_rejected(tmp_path, change, message):
    path, manifest = fixture_manifest(tmp_path)
    manifest["rows"][0].update(change)
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=message):
        load_prepared_manifest(path, ["avqa"])


def test_changed_sampling_rejected(tmp_path):
    path, _ = fixture_manifest(tmp_path)
    with pytest.raises(ValueError, match="source pattern"):
        load_prepared_manifest(path, ["avqa", "avqa"])
