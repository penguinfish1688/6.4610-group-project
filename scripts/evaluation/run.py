"""Run paired AVQA ablations from a release or an external prepared manifest."""

import argparse
import importlib.metadata
import json
from pathlib import Path

from scripts.evaluation.report import (
    CONDITIONS, digest, donor_map, guard_run, report, select_rows,
)


def run(args):
    bundle = args.bundle.resolve()
    prepared_root = getattr(args, "prepared_dataset", None)
    if prepared_root:
        if args.manifest or args.data_root:
            raise ValueError("--prepared-dataset cannot be combined with --manifest/--data-root")
        from scripts.evaluation.prepared import inventory

        data_root = prepared_root.resolve()
        manifest_path = data_root / "manifest.json"
        _, prepared_rows = inventory(data_root)
        manifest = dict(cases=prepared_rows)
    else:
        manifest_path = (args.manifest or bundle / "examples/manifest.json").resolve()
        data_root = (args.data_root or bundle).resolve()
        manifest = json.loads(manifest_path.read_text())
    rows = select_rows(manifest, args.cases, args.limit)
    if len(set(args.conditions)) != len(args.conditions):
        raise ValueError("Duplicate conditions")
    variant = getattr(args, "model", "finetuned")
    if variant == "base-d2" and args.conditions != ["padding"]:
        raise ValueError("Original D2 has no visual lane; use --model base-d2 --conditions padding")
    donors = donor_map(rows) if "mismatched" in args.conditions else {}
    cfg = json.loads((bundle / "inference-config.json").read_text())
    # The loader verifies input SHA256 and native prompts before any rollout.
    for row in rows:
        path = (data_root / row["input"]).resolve()
        if not path.is_relative_to(data_root) or not path.is_file():
            raise ValueError(f"Missing or unsafe prepared input: {path}")
    plan = [dict(case_id=row["name"], condition=condition,
                 donor_id=donors.get(row["name"]) if condition == "mismatched" else None)
            for row in rows for condition in args.conditions]
    identity = dict(
        schema="d2_omni.evaluation.v2", bundle=str(bundle), data_root=str(data_root),
        model=variant, input_mode="prepared_mel" if prepared_root else "waveform",
        manifest_sha256=digest(manifest_path), seed=cfg["seed"], plan=plan,
        config_sha256=digest(bundle / "inference-config.json"),
        checkpoint_manifest_sha256=digest(bundle / "finetuned/latest.json"),
        runtime_versions={name: importlib.metadata.version(name) for name in (
            "torch", "transformers", "safetensors",
        )} if not args.dry_run else {},
        code_sha256={},
    )
    root = Path(__file__).resolve().parents[2]
    for name in ("scripts/evaluation/__main__.py", "scripts/evaluation/run.py",
                 "scripts/evaluation/report.py", "scripts/evaluation/inputs.py",
                 "scripts/evaluation/prepared.py",
                 "scripts/infer_omni.py", "model/release_bundle.py", "model/int8_runtime.py",
                 "scripts/quantize_experts.py", "scripts/inference_checkpoint.py",
                 "model/backbone/qwen3-omni-D2/inference/session.py",
                 "model/backbone/qwen3-omni-D2/model/duplex.py"):
        identity["code_sha256"][name] = digest(root / name)
    if prepared_root:
        identity["split_catalog_sha256"] = {
            split: digest(data_root / f"{split}.jsonl") for split in ("train", "probe")
        }
    print(json.dumps(dict(rollouts=len(plan), plan=plan), indent=2), flush=True)
    if args.dry_run:
        return
    guard_run(args.output, identity)
    # GPU/D2 imports are deferred so --help and --dry-run need no model runtime.
    from model.release_bundle import select_cases
    from scripts.evaluation.inputs import ablate_visual
    from scripts.infer_omni import run as infer

    def load_cases(_config, processor):
        if prepared_root:
            from scripts.evaluation.prepared import select_cases as prepared_cases

            cases = prepared_cases(data_root, rows, processor.tokenizer)
        else:
            cases = select_cases(
                bundle, processor.tokenizer, manifest_path=manifest_path, data_root=data_root,
                names={row["name"] for row in rows},
            )
        indexed = {case["name"]: case for case in cases}
        expanded = []
        for item in plan:
            case = indexed[item["case_id"]]
            donor = indexed[item["donor_id"]] if item["donor_id"] else None
            expanded.append(dict(
                case, name=f"{case['name']}--{item['condition']}", evaluation=item,
                visual=ablate_visual(case["visual"], item["condition"],
                                    donor["visual"] if donor else None),
                video_path=(None if item["condition"] == "padding" else
                            donor["video_path"] if donor else case["video_path"]),
            ))
        return expanded

    infer(args, case_loader=load_cases, resume=True,
          on_result=lambda results: report(args.output, results, len(plan)))
    print(f"Evaluation report: {args.output / 'evaluation.html'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=Path("weight/base_1030"))
    parser.add_argument("--manifest", type=Path, help="External manifest using the release case schema")
    parser.add_argument("--data-root", type=Path, help="Root for manifest input/video paths; default: bundle")
    parser.add_argument("--prepared-dataset", type=Path, help="Downloaded step9000 training-data release")
    parser.add_argument("--model", choices=("finetuned", "base-d2"), default="finetuned")
    parser.add_argument("--cases", nargs="+", help="AVQA case names; default: all AVQA cases")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument("--output", type=Path, default=Path("runs/avqa-evaluation"))
    parser.add_argument("--dry-run", action="store_true", help="Preview the plan without loading the model")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
