"""Load a portable, checksummed manifest without private source-data loaders."""

import hashlib
import json
from pathlib import Path


def load_prepared_manifest(path, source_pattern):
    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    if manifest.get("format") != "d2_omni.dataset.v1":
        raise ValueError("Unsupported prepared dataset manifest")
    if manifest["source_pattern"] != source_pattern:
        raise ValueError("Dataset source pattern differs from training config")
    if manifest.get("official_test_used_for_training") is not False:
        raise ValueError("Dataset must exclude official test media")
    rows, seen = [], set()
    for row in manifest["rows"]:
        relative = Path(row["path"])
        target = (path.parent / relative).resolve()
        if relative.is_absolute() or not target.is_relative_to(path.parent):
            raise ValueError("Prepared sample escapes dataset directory")
        if target in seen:
            raise ValueError("Duplicate prepared sample")
        seen.add(target)
        if row["source"] not in source_pattern or row["split"] not in {"train", "probe"}:
            raise ValueError("Unknown dataset source or split")
        if not isinstance(row["ticks"], int) or row["ticks"] < 9000:
            raise ValueError("Prepared conversation is shorter than 720 seconds")
        if target.stat().st_size != row["bytes"]:
            raise ValueError(f"Prepared sample size mismatch: {relative}")
        with target.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != row["sha256"]:
            raise ValueError(f"Prepared sample checksum mismatch: {relative}")
        # Preserve order: rank assignment and cyclic sampling depend on it.
        rows.append(dict(row, path=str(target)))
    if not rows:
        raise ValueError("Empty prepared dataset")
    return rows
