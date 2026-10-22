"""Fetch a small raw-video AVQA training subset, retaining official splits."""

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time

import requests


def main():
    root = Path("data/raw/avqa")
    output = root / "videos"
    output.mkdir(parents=True, exist_ok=True)
    train = json.loads((root / "annotations/train_qa.json").read_text())
    test = json.loads((root / "annotations/val_qa.json").read_text())
    train_names = {row["video_name"] for row in train}
    test_names = {row["video_name"] for row in test}
    repository = "juyil/AVQA-videos"
    info = requests.get(f"https://huggingface.co/api/datasets/{repository}", timeout=40)
    info.raise_for_status()
    revision = info.json()["sha"]
    tree = requests.get(
        f"https://huggingface.co/api/datasets/{repository}/tree/{revision}/videos", timeout=40
    )
    tree.raise_for_status()
    available = {Path(item["path"]).stem: item for item in tree.json() if item["type"] == "file"}
    audio_root = Path(
        "/data/sls/scratch/chang168/LSKV-runtime/orcd/scratch/orcd/011/chang168/dataset/avqa_small/sft_async_v3"
    )
    manifest = json.loads((audio_root / "manifest.json").read_text())
    selected = {}
    for address in manifest["rows"]:
        row = json.loads((audio_root / address["row_path"]).read_text())
        name = Path(row["file_name"]).stem
        if name in available and name in train_names and name not in test_names:
            selected.setdefault(name, available[name])
        if len(selected) >= 384:
            break
    if len(selected) < 8:
        raise RuntimeError("Too few matching official-train raw videos in the public mirror")
    print(
        json.dumps(
            dict(
                event="selected",
                videos=len(selected),
                repository=repository,
                revision=revision,
                official_test_excluded=True,
            )
        ),
        flush=True,
    )

    def download(item):
        name, descriptor = item
        path = output / f"{name}.mp4"
        expected = descriptor.get("lfs", {}).get("oid")
        if path.exists() and path.stat().st_size == descriptor["size"]:
            return dict(video=name, bytes=path.stat().st_size, reused=True)
        response = requests.get(
            f"https://huggingface.co/datasets/{repository}/resolve/{revision}/{descriptor['path']}",
            stream=True,
            timeout=(20, 120),
        )
        response.raise_for_status()
        temporary = path.with_suffix(".partial")
        digest = hashlib.sha256()
        with temporary.open("wb") as stream:
            for chunk in response.iter_content(1024 * 1024):
                if chunk:
                    stream.write(chunk)
                    digest.update(chunk)
        response.close()
        if temporary.stat().st_size != descriptor["size"] or (
            expected and digest.hexdigest() != expected
        ):
            raise ValueError(f"Raw-video mirror checksum mismatch: {name}")
        temporary.replace(path)
        return dict(
            video=name,
            bytes=path.stat().st_size,
            sha256=digest.hexdigest(),
            official_split="train",
            repository=repository,
            revision=revision,
        )

    receipts, failures = [], []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(download, item): item[0] for item in selected.items()}
        for future in as_completed(futures):
            try:
                receipt = future.result()
                receipts.append(receipt)
                print(json.dumps(dict(event="downloaded", **receipt)), flush=True)
            except Exception as error:
                failures.append(dict(video=futures[future], error=str(error)))
                print(json.dumps(dict(event="download_failed", **failures[-1])), flush=True)
    (root / "video-download-provenance.json").write_text(
        json.dumps(
            dict(
                repository=repository,
                revision=revision,
                downloaded_at=time.time(),
                videos=receipts,
                failures=failures,
                official_test_used=False,
            ),
            indent=2,
        )
        + "\n"
    )
    if len(receipts) < 8:
        raise RuntimeError("Raw-video acquisition did not produce enough usable AVQA clips")


if __name__ == "__main__":
    main()
