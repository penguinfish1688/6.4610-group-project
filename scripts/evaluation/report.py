"""CPU-only planning, answer scoring, and reports for paired AVQA evaluation."""

import csv
import hashlib
import html
import json
from pathlib import Path
import re

CONDITIONS = ("correct", "padding", "mismatched")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def parse_choice(text, choices="ABCD"):
    """Accept a leading choice, optionally prefixed by 'Answer:' or 'The answer is'.

    Do not search arbitrary prose for letters or infer answers from option wording.
    """
    match = re.match(
        r"^\s*(?:(?:the\s+)?answer\s*(?:is\s*|:\s*))?"
        r"\(?([A-Z])\)?(?=\s|[.:,;\-]|$)", text, re.IGNORECASE,
    )
    value = match[1].upper() if match else None
    return value if value is not None and value in choices else None


def select_rows(manifest, names=None, limit=None):
    rows = [row for row in manifest["cases"] if row["task"] == "avqa"]
    if names:
        missing = set(names) - {row["name"] for row in rows}
        if missing:
            raise ValueError(f"Unknown AVQA cases: {sorted(missing)}")
        rows = [row for row in rows if row["name"] in names]
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be positive")
        rows = rows[:limit]
    if not rows:
        raise ValueError("No AVQA cases selected")
    seen = set()
    for row in rows:
        name = row["name"]
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name) or name in seen:
            raise ValueError(f"Unsafe or duplicate case name: {name}")
        seen.add(name)
        if not row.get("reference_from_targets") and parse_choice(row["reference"]) is None:
            raise ValueError(f"Reference must begin with a choice letter: {name}")
    return rows


def donor_map(rows):
    """Choose the next distinct video in manifest order, deterministically."""
    donors = {}
    for i, row in enumerate(rows):
        source = row.get("source_video_id") or row.get("video")
        if not source:
            raise ValueError(f"Missing video identity for {row['name']}")
        for offset in range(1, len(rows)):
            other = rows[(i + offset) % len(rows)]
            identity = other.get("source_video_id") or other.get("video")
            if identity and identity != source:
                donors[row["name"]] = other["name"]
                break
        else:
            raise ValueError("Mismatched condition needs at least two distinct videos")
    return donors


def guard_run(output, identity):
    """Refuse to mix results from different manifests, settings or code."""
    output = Path(output)
    path = output / "evaluation-run.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != identity:
            raise ValueError("Evaluation settings changed; choose a new --output directory")
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError("Output is not empty and has no evaluation-run.json")
        output.mkdir(parents=True, exist_ok=True)
        write_json(path, identity)


def report(output, results, planned):
    rows = []
    for result in results:
        metadata = result["evaluation"]
        expected = parse_choice(result["reference_text"])
        predicted = parse_choice(result["generated_text"])
        rows.append(dict(
            case_id=metadata["case_id"], condition=metadata["condition"],
            donor_id=metadata.get("donor_id"), reference=expected, prediction=predicted,
            correct=predicted == expected, invalid=predicted is None,
            generated_text=result["generated_text"], reference_text=result["reference_text"],
            audio_seconds=result["audio_seconds"], wall_seconds=result["wall_seconds"],
            visual_tokens=result["visual_tokens"],
            active_speech_seconds=0.08 * sum(
                bool(event.get("metadata", {}).get("emit_audio"))
                for event in result.get("events", [])
            ),
            output_audio=f"{result['name']}/output.wav",
        ))
    conditions = {}
    for condition in CONDITIONS:
        group = [row for row in rows if row["condition"] == condition]
        if group:
            conditions[condition] = dict(
                completed=len(group), correct=sum(row["correct"] for row in group),
                invalid=sum(row["invalid"] for row in group),
                accuracy=sum(row["correct"] for row in group) / len(group),
                real_time_factor=sum(row["wall_seconds"] for row in group)
                / sum(row["audio_seconds"] for row in group),
            )
    indexed = {(row["case_id"], row["condition"]): row for row in rows}
    paired = {}
    for condition in ("padding", "mismatched"):
        pairs = [(row, indexed[(row["case_id"], condition)]) for row in rows
                 if row["condition"] == "correct" and (row["case_id"], condition) in indexed]
        if pairs:
            paired[condition] = dict(
                completed_pairs=len(pairs),
                correct_video_only_wins=sum(a["correct"] and not b["correct"] for a, b in pairs),
                ablation_only_wins=sum(b["correct"] and not a["correct"] for a, b in pairs),
                accuracy_difference=sum(int(a["correct"]) - int(b["correct"])
                                        for a, b in pairs) / len(pairs),
            )
    summary = dict(
        planned=planned, completed=len(rows), complete=len(rows) == planned,
        model_variants=sorted({result.get("model_variant", "finetuned") for result in results}),
        conditions=conditions, paired=paired, examples=rows,
        timing_note="Cached visual inputs; excludes model loading and vision encoding. "
                    "Active speech duration is not a speech-completion score.",
    )
    output = Path(output)
    write_json(output / "evaluation.json", summary)
    if rows:
        with (output / "evaluation.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    cards = []
    for row in rows:
        cards.append(
            f"<tr><td>{html.escape(row['case_id'])}</td><td>{row['condition']}</td>"
            f"<td>{html.escape(row['reference_text'])}</td>"
            f"<td>{html.escape(row['generated_text'])}</td><td>{row['correct']}</td>"
            f"<td><audio controls preload='none' src='{row['output_audio']}'></audio></td></tr>"
        )
    (output / "evaluation.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>AVQA evaluation</title>"
        "<style>body{font:16px system-ui;margin:24px}td,th{padding:10px;border:1px solid #ccc}"
        "table{border-collapse:collapse}</style><h1>AVQA evaluation</h1>"
        f"<p>Completed {len(rows)} / {planned} rollouts. Small demo sets are debugging checks.</p>"
        f"<pre>{html.escape(json.dumps(conditions, indent=2))}</pre>"
        "<table><tr><th>Case</th><th>Condition</th><th>Reference</th><th>Generated</th>"
        "<th>Correct</th><th>Speech</th></tr>" + "".join(cards) + "</table>",
        encoding="utf-8",
    )
    return summary
