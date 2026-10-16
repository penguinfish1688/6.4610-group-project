# Progress updates

Use `YYYY-MM-DD — name`, followed by what changed, evidence/artifacts, and the
next step. Distinguish implemented behavior from planned experiments.

## 2026-10-09 — CK (from the initial project notes)

- Proposed extending D2 with streaming vision.
- Suggested running D2 on ORCD first, using a GPU with more than 70 GB VRAM.
- Invited teammates to clarify the writing and claim project tasks.

## 2026-10-16 — Codex, repository setup requested by CK

- Created the D2-Omni repository layout and imported D2 source at the revision
  recorded in `model/backbone/UPSTREAM.json`.
- Added editable native Qwen3-Omni vision code, a local checkpoint export utility,
  and small CPU tests against the pinned upstream implementation.
- Added draft timestamped data examples, research settings, task ownership,
  and evaluation/demo plans.
- Validation: tiny CPU models matched upstream outputs and parameter gradients
  with eager and SDPA attention. Vision-only checkpoint export/reload passed for
  single-file and sharded snapshots. All 117 imported D2 files matched upstream.
- Next: profile the native vision path and define the first Stage II sample.

## Template

```text
YYYY-MM-DD — name
Completed:
Evidence / artifacts:
Open issues:
Next:
```
