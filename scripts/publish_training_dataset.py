"""Upload a prepared dataset release and verify every remote file's digest."""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--repo", default="penguinfish1688/D2-Omni-step9000-training-data")
    parser.add_argument("--token-file", type=Path)
    parser.add_argument("--receipt", type=Path, default=Path(".cache/dataset-upload-receipt.json"))
    args = parser.parse_args()
    from huggingface_hub import HfApi, hf_hub_download
    from scripts.download_release import verify

    verify(args.folder)
    token = args.token_file.read_text().strip() if args.token_file else None
    api = HfApi(token=token)
    api.create_repo(repo_id=args.repo, repo_type="dataset", private=False, exist_ok=True)
    api.upload_folder(
        repo_id=args.repo,
        repo_type="dataset",
        folder_path=args.folder,
        ignore_patterns=[".cache/**", "**/.cache/**"],
        commit_message="Release exact step-9000 corpus, training recipe and audio pipeline",
    )
    public = HfApi(token=False)
    info = public.dataset_info(args.repo, files_metadata=True)
    if info.private:
        raise ValueError("Dataset is not public")
    remote = {item.rfilename: item for item in info.siblings}
    expected = {}
    for line in (args.folder / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1)
        expected[name] = digest
    expected["SHA256SUMS"] = hashlib.sha256((args.folder / "SHA256SUMS").read_bytes()).hexdigest()
    total = 0
    for name, digest in expected.items():
        local = args.folder / name
        entry = remote[name]
        if entry.size != local.stat().st_size:
            raise ValueError(f"Remote size mismatch: {name}")
        if entry.lfs is not None:
            actual = entry.lfs.sha256
        else:
            downloaded = hf_hub_download(
                args.repo,
                name,
                repo_type="dataset",
                revision=info.sha,
                token=False,
                cache_dir=".cache/hf-dataset-verify",
            )
            with open(downloaded, "rb") as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != digest:
            raise ValueError(f"Remote digest mismatch: {name}")
        total += entry.size
    receipt = dict(
        repo_id=args.repo,
        revision=info.sha,
        public=True,
        verified_files=len(expected),
        verified_bytes=total,
    )
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
