"""Prune superseded rank shards after a complete checkpoint is published."""

import json
from pathlib import Path
import re
import shutil


_GENERATION = re.compile(r"step-(\d{8})-(?:\d+|local)")


def checkpoint_retention_policy(config):
    policy = config.get("checkpoint_retention", "all")
    if policy not in ("all", "latest_only"):
        raise ValueError("checkpoint_retention must be 'all' or 'latest_only'")
    return policy


def prune_checkpoints(output, manifest):
    """Keep the published complete generation; leave unrelated paths untouched."""
    output = Path(output)
    latest = output / "latest.json"
    if json.loads(latest.read_text()) != manifest:
        raise ValueError("Published checkpoint changed; refusing to prune")
    shards = manifest["shards"]
    if not shards or len(shards) != manifest["world_size"]:
        raise ValueError("Published checkpoint shard inventory is incomplete")
    protected, ranks = set(), set()
    for entry in shards:
        relative = Path(entry["path"])
        if relative.is_absolute() or len(relative.parts) != 2:
            raise ValueError("Published checkpoint shard path is invalid")
        generation = _GENERATION.fullmatch(relative.parts[0])
        if generation is None or int(generation[1]) != manifest["step"]:
            raise ValueError("Published checkpoint generation does not match its step")
        rank = re.fullmatch(r"rank-(\d{3})\.pt", relative.name)
        if rank is None:
            raise ValueError("Published checkpoint rank filename is invalid")
        ranks.add(int(rank[1]))
        folder = output / relative.parts[0]
        path = output / relative
        if folder.is_symlink() or path.is_symlink():
            raise ValueError("Published checkpoint must contain local rank files")
        if not path.is_file() or path.stat().st_size != entry["bytes"]:
            raise ValueError("Published checkpoint shard is missing or incomplete")
        protected.add(folder.name)
    if len(protected) != 1 or ranks != set(range(manifest["world_size"])):
        raise ValueError("Published checkpoint has mixed generations or duplicate ranks")
    removed = []
    for folder in sorted(output.iterdir()):
        generation = _GENERATION.fullmatch(folder.name)
        if (
            generation is None
            or folder.name in protected
            or int(generation[1]) > manifest["step"]
            or folder.is_symlink()
            or not folder.is_dir()
        ):
            continue
        if json.loads(latest.read_text()) != manifest:
            raise ValueError("Published checkpoint changed; refusing to prune")
        allocated = sum(
            path.stat().st_blocks * 512
            for path in folder.rglob("*")
            if not path.is_symlink() and path.is_file()
        )
        shutil.rmtree(folder)
        removed.append(dict(path=folder.name, allocated_bytes=allocated))
    return dict(
        step=manifest["step"], policy="latest_only", kept=sorted(protected), removed=removed
    )
