"""Download the public inference bundle into this repository's weight directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def verify(folder):
    rows = (folder / "SHA256SUMS").read_text().splitlines()
    for row in rows:
        expected, name = row.split("  ", 1)
        path = (folder / name).resolve()
        if not path.is_relative_to(folder.resolve()):
            raise ValueError("Invalid checksum path")
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected:
            raise ValueError(f"Checksum mismatch: {name}")
    print(f"Verified {len(rows)} files in {folder}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "weight/base_1030")
    parser.add_argument("--revision", help="Override the recorded Hugging Face commit")
    parser.add_argument("--verify", action="store_true", help="Read and SHA256-check all downloaded files")
    args = parser.parse_args()
    pin = json.loads((ROOT / "weight/base_1030/hub.json").read_text())
    os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface-download"))
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=pin["repo_id"], revision=args.revision or pin["revision"],
        local_dir=args.output, token=False, max_workers=4,
    )
    if args.verify:
        verify(args.output)
    print(f"Ready: {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
