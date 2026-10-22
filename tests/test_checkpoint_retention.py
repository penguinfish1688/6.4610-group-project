"""Retain the published checkpoint without touching incomplete or unrelated data."""

from datetime import timedelta
import json
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from scripts.checkpoint_retention import checkpoint_retention_policy, prune_checkpoints


def publish(root, step=20, job=2):
    folder = root / f"step-{step:08d}-{job}"
    folder.mkdir()
    entries = []
    for rank in range(2):
        path = folder / f"rank-{rank:03d}.pt"
        path.write_bytes(b"complete")
        entries.append(dict(path=str(path.relative_to(root)), bytes=path.stat().st_size))
    manifest = dict(step=step, world_size=2, shards=entries)
    (root / "latest.json").write_text(json.dumps(manifest))
    return folder, manifest


def test_only_published_generation_survives(tmp_path):
    old, _ = publish(tmp_path, 10, 1)
    same_step, _ = publish(tmp_path, 20, 1)
    latest, manifest = publish(tmp_path)
    future = tmp_path / "step-00000021-3"
    future.mkdir()
    (future / "rank-000.partial").write_bytes(b"unfinished")
    unrelated = tmp_path / "data"
    unrelated.mkdir()
    (unrelated / "sample.pt").write_bytes(b"dataset")
    report = prune_checkpoints(tmp_path, manifest)
    assert not old.exists() and not same_step.exists()
    assert latest.is_dir() and future.is_dir() and unrelated.is_dir()
    assert {row["path"] for row in report["removed"]} == {old.name, same_step.name}
    assert json.loads((tmp_path / "latest.json").read_text()) == manifest
    assert prune_checkpoints(tmp_path, manifest)["removed"] == []


@pytest.mark.parametrize("failure", ["missing", "truncated", "duplicate", "escaped", "stale"])
def test_invalid_publication_cannot_delete_older_checkpoint(tmp_path, failure):
    old, _ = publish(tmp_path, 10, 1)
    latest, manifest = publish(tmp_path)
    if failure == "missing":
        (latest / "rank-001.pt").unlink()
    elif failure == "truncated":
        (latest / "rank-001.pt").write_bytes(b"x")
    elif failure == "duplicate":
        manifest["shards"][1] = manifest["shards"][0]
    elif failure == "escaped":
        manifest["shards"][0]["path"] = "../outside.pt"
    if failure != "stale":
        (tmp_path / "latest.json").write_text(json.dumps(manifest))
    else:
        (tmp_path / "latest.json").write_text("{}")
    with pytest.raises(ValueError):
        prune_checkpoints(tmp_path, manifest)
    assert old.is_dir() and latest.is_dir()


def test_symlinks_are_not_traversed_or_removed(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    outside = tmp_path / "shared-dataset"
    outside.mkdir()
    (outside / "sample.pt").write_bytes(b"keep")
    link = run / "step-00000010-1"
    link.symlink_to(outside, target_is_directory=True)
    _, manifest = publish(run)
    assert prune_checkpoints(run, manifest)["removed"] == []
    assert link.is_symlink() and (outside / "sample.pt").read_bytes() == b"keep"


def test_symlinked_latest_shard_cannot_authorize_pruning(tmp_path):
    old, _ = publish(tmp_path, 10, 1)
    latest, manifest = publish(tmp_path)
    path = latest / "rank-001.pt"
    path.unlink()
    path.symlink_to(latest / "rank-000.pt")
    with pytest.raises(ValueError, match="local rank files"):
        prune_checkpoints(tmp_path, manifest)
    assert old.is_dir()


def test_policy_defaults_preserve_legacy_recipe():
    assert checkpoint_retention_policy({}) == "all"
    assert checkpoint_retention_policy(dict(checkpoint_retention="latest_only")) == "latest_only"
    with pytest.raises(ValueError):
        checkpoint_retention_policy(dict(checkpoint_retention="invalid"))


def _save_two_generations(rank, rendezvous, output):
    from scripts.distributed import read_checkpoint, restore_checkpoint, save_checkpoint

    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", rank=rank, world_size=2, init_method=rendezvous, timeout=timedelta(seconds=30)
    )
    try:
        parameter = torch.nn.Parameter(torch.tensor([float(rank)]))
        optimizer = torch.optim.AdamW([parameter])
        for step in (1, 2):
            parameter.grad = torch.ones_like(parameter)
            optimizer.step()
            save_checkpoint(
                output,
                {"adapter": parameter},
                optimizer,
                dict(step=step, config=dict(checkpoint_retention="latest_only")),
            )
            if step == 1:
                payload = read_checkpoint(output)
                restore_checkpoint(payload, {"adapter": parameter}, optimizer)
                for key, original in payload["optimizer_states"]["adapter"].items():
                    if torch.is_tensor(original):
                        restored = optimizer.state[parameter][key]
                        torch.testing.assert_close(restored, original)
                        assert (
                            restored.untyped_storage().data_ptr()
                            != original.untyped_storage().data_ptr()
                        )
                del original, restored, payload
        assert len(list(Path(output).glob("step-*"))) == 1
    finally:
        dist.destroy_process_group()


def test_distributed_save_prunes_only_after_all_shards_are_published(tmp_path):
    output = tmp_path / "run"
    rendezvous = (tmp_path / "rendezvous").as_uri()
    mp.spawn(_save_two_generations, args=(rendezvous, str(output)), nprocs=2, join=True)
    manifest = json.loads((output / "latest.json").read_text())
    assert manifest["step"] == 2 and len(manifest["shards"]) == 2
    for entry in manifest["shards"]:
        assert (output / entry["path"]).stat().st_size == entry["bytes"]
    retention = json.loads((output / "retention-latest.json").read_text())
    assert retention["step"] == 2 and len(retention["removed"]) == 1
