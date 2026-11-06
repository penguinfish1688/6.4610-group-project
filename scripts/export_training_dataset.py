"""Package the exact step-9000 prepared corpus; never re-materialize examples."""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile


ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def export(output):
    import torch

    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Use a new output directory: {output}")
    (output / "prepared").mkdir(parents=True)
    run = ROOT / "runs/d2-omni-xl-80ms"
    original = json.loads((run / "data_manifest.json").read_text())
    rows, catalog = [], []
    counts, durations = Counter(), Counter()
    clips = {"train": set(), "probe": set()}
    for row in original["rows"]:
        source = ROOT / row["path"]
        payload = torch.load(source, map_location="cpu", mmap=True, weights_only=True)
        assert payload["format"] == "d2_omni.prepared.xl.v1"
        assert all(payload[key] == row[key] for key in ("ticks", "source", "split"))
        assert row["ticks"] >= 9000
        tensors = payload["tensors"]
        assert tensors["env_mel"].shape[-1] == row["ticks"] * 8
        assert tensors["self_mel"].shape[-1] == row["ticks"] * 8
        assert tensors["text_target"].shape == (1, row["ticks"])
        assert tensors["codec_target"].shape == (1, row["ticks"], 16)
        target = output / "prepared" / source.name
        try:
            os.link(source, target)
        except OSError:
            shutil.copyfile(source, target)
        record = dict(
            row,
            path=target.relative_to(output).as_posix(),
            bytes=target.stat().st_size,
            sha256=digest(target),
        )
        rows.append(record)
        detail = dict(
            record,
            original_path=row["path"],
            system_prompt_input_ids=tensors["system_prompt_input_ids"][0].tolist(),
            tensors={
                key: dict(shape=list(value.shape), dtype=str(value.dtype))
                for key, value in tensors.items()
            },
        )
        for key in ("source_id", "source_members", "placeholder_seed", "official_split"):
            if key in payload:
                detail[key] = payload[key]
        if row["source"] == "avqa":
            clips[row["split"]].update(payload["source_members"])
        catalog.append(detail)
        key = row["source"] + "/" + row["split"]
        counts[key] += 1
        durations[key] += row["ticks"] * 0.08
        print(f"Verified {len(rows)}/{len(original['rows'])}: {source.name}", flush=True)
    assert not clips["train"] & clips["probe"]

    def clip_key(name):
        video, second = Path(name).stem.rsplit("_", 1)
        return f"{video}_{int(second):06d}"

    official = json.loads((ROOT / "data/raw/avqa/annotations/val_qa.json").read_text())
    official_clips = {clip_key(row["video_name"]) for row in official}
    assert not (clips["train"] | clips["probe"]) & official_clips
    write_json(
        output / "manifest.json",
        dict(
            format="d2_omni.dataset.v1",
            checkpoint_step=9000,
            rows=rows,
            source_pattern=original["source_pattern"],
            official_test_used_for_training=False,
        ),
    )
    for split in ("train", "probe"):
        (output / f"{split}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in catalog if row["split"] == split)
        )
    write_json(
        output / "provenance/release_audit.json",
        dict(
            checks_passed=True,
            conversation_counts=dict(counts),
            seconds=dict(durations),
            avqa_unique_clips={key: len(value) for key, value in clips.items()},
            official_test_overlap=0,
            train_probe_avqa_overlap=0,
            prepared_bytes=sum(row["bytes"] for row in rows),
        ),
    )
    for name in (
        "data_manifest.json",
        "prepared_data_audit.json",
        "latest.json",
        "launch-96812.json",
        "launch-96843.json",
        "stop-96843.json",
    ):
        shutil.copyfile(run / name, output / "provenance" / name)
    prepared = ROOT / "data/processed/native-xl-80ms"
    for path in [prepared / "audio_recipe.json", *sorted(prepared.glob("exclusions-rank*.json"))]:
        shutil.copyfile(path, output / "provenance" / path.name)
    for name in ("README.md", "TRAINING.md"):
        shutil.copyfile(ROOT / "data/releases/step9000" / name, output / name)
    pipeline = (
        (ROOT / "docs/data-pipeline.md")
        .read_text()
        .replace("../data/releases/step9000/TRAINING.md", "TRAINING.md")
    )
    (output / "DATA_PIPELINE.md").write_text(pipeline)
    (output / "configs").mkdir()
    for name in ("train-step9000-phase1.json", "train-step9000-phase2.json"):
        shutil.copyfile(ROOT / "configs" / name, output / "configs" / name)
    shutil.copyfile(ROOT / "configs/train-sls.json", output / "provenance/train-sls.json")
    shutil.copyfile(
        ROOT / "model/backbone/qwen3-omni-D2/configs/train.json",
        output / "provenance/optimizer.json",
    )
    # Explicit source allowlist excludes local secrets, caches, media and weights.
    files = (
        subprocess.check_output(
            ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"], cwd=ROOT
        )
        .decode()
        .split("\0")
    )
    allowed = (
        "scripts/",
        "model/",
        "tests/",
        "configs/",
        "docs/",
        "data/examples/",
        "data/releases/step9000/",
        "weight/base_1030/",
    )
    source_files = sorted(
        {
            name
            for name in files
            if name
            and (ROOT / name).is_file()
            and (name.startswith(allowed) or name in {"README.md", "pyproject.toml"})
            and (
                Path(name).suffix
                in {
                    ".py",
                    ".json",
                    ".jsonl",
                    ".toml",
                    ".md",
                    ".txt",
                    ".sh",
                    ".sbatch",
                    ".html",
                    ".js",
                }
                or Path(name).name == "LICENSE"
            )
        }
    )
    with tarfile.open(output / "source.tar.gz", "w:gz") as archive:
        for name in source_files:
            archive.add(ROOT / name, arcname="D2-Omni/" + name, recursive=False)
    write_json(
        output / "provenance/source.json",
        dict(
            repository="https://github.com/penguinfish1688/6.4610-group-project",
            base_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT)
            .decode()
            .strip(),
            note="Working source snapshot includes the portable prepared-manifest loader.",
            files={name: digest(ROOT / name) for name in source_files},
        ),
    )
    sums = []
    for path in sorted(output.rglob("*")):
        if path.is_file():
            sums.append(f"{digest(path)}  {path.relative_to(output).as_posix()}\n")
    (output / "SHA256SUMS").write_text("".join(sums))
    print(
        json.dumps(
            dict(
                output=str(output),
                files=len(sums),
                prepared_files=len(rows),
                prepared_bytes=sum(row["bytes"] for row in rows),
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / ".cache/dataset-step9000")
    export(parser.parse_args().output)
