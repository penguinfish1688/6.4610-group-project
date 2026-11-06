"""Download and verify the pinned AVQA probe tensors and split inventory."""

import argparse
import hashlib
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/processed/step9000-release"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    pin = json.loads((root / "data/releases/step9000/hub.json").read_text())
    os.environ.setdefault("HF_HOME", str(root / ".cache/huggingface-download"))
    from huggingface_hub import hf_hub_download

    def download(name):
        return Path(hf_hub_download(
            pin["repo_id"], name, repo_type="dataset", revision=pin["revision"],
            local_dir=args.output, token=False,
        ))

    for name in ("manifest.json", "train.jsonl", "probe.jsonl",
                 "provenance/release_audit.json", "SHA256SUMS"):
        download(name)
    manifest = json.loads((args.output / "manifest.json").read_text())
    rows = [row for row in manifest["rows"]
            if row["source"] == "avqa" and row["split"] == "probe"]
    if len(rows) != 4 or manifest.get("official_test_used_for_training") is not False:
        raise ValueError("Unexpected pinned release inventory")
    for row in rows:
        path = download(row["path"])
        with path.open("rb") as stream:
            checksum = hashlib.file_digest(stream, "sha256").hexdigest()
        if path.stat().st_size != row["bytes"] or checksum != row["sha256"]:
            raise ValueError(f"Checksum/size mismatch: {path}")
        print(f"Verified {row['path']}", flush=True)
    print(f"Ready: {args.output.resolve()} ({sum(r['bytes'] for r in rows):,} tensor bytes)")


if __name__ == "__main__":
    main()
