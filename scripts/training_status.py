"""Summarize actual GPU updates and fixed validation losses from the run log."""

import argparse
import json
from pathlib import Path
import statistics


def summarize(root):
    events = []
    for line in (root / "metrics-rank0.jsonl").read_text().splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    latest_start = max(
        (i for i, event in enumerate(events) if event["event"] == "starting"), default=0
    )
    current = events[latest_start:]
    updates = [event for event in current if event["event"] == "update"]
    probes = [event for event in current if event["event"] == "probe"]
    result = dict(last_event=events[-1] if events else None, updates=len(updates))
    if updates:
        result.update(
            last_update=updates[-1],
            median_step_seconds=statistics.median(e["seconds"] for e in updates[-20:]),
            per_source={
                source: dict(
                    updates=len(rows),
                    last_loss=rows[-1]["loss"],
                    recent_mean_loss=statistics.mean(row["loss"] for row in rows[-20:]),
                )
                for source in sorted({e["source"] for e in updates})
                if (rows := [e for e in updates if e["source"] == source])
            },
        )
    if probes:
        result.update(
            first_probe=probes[0],
            latest_probe=probes[-1],
            fixed_probe_loss_change=probes[-1]["loss"] - probes[0]["loss"],
        )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("runs/d2-omni-xl-80ms"))
    args = parser.parse_args()
    print(json.dumps(summarize(args.run), indent=2))
