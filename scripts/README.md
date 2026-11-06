# Project scripts

Run Python commands from the repository root with `python -m ...` so imports
resolve consistently on local machines and cluster nodes.

## Evaluation

Evaluation-specific code lives in `scripts/evaluation/`:

```text
evaluation/
  __main__.py     Entry point: python -m scripts.evaluation
  run.py          Select inputs, plan conditions, validate resume identity
  inputs.py       Correct, padded, and mismatched visual inputs
  report.py       Choice scoring, paired summaries, HTML/CSV/JSON reports
  orcd.sbatch     ORCD GPU batch job
```

The evaluator reuses `scripts/infer_omni.py` for model loading and free-running
generation. Shared inference and checkpoint utilities remain at the scripts
root because they also support the original demo and training workflows.
The protocol, dataset format, and ORCD commands are in
[evaluation/RUNNER.md](../evaluation/RUNNER.md).

```bash
python -m scripts.evaluation --dry-run
mkdir -p logs
sbatch scripts/evaluation/orcd.sbatch --output runs/avqa-eval-demo-v2
```

## Other entry points

| Workflow | Entry points |
| --- | --- |
| Release download/export | `download_release.py`, `export_inference_release.py` |
| Dataset preparation | `prepare_native_data.py`, `prepare_avqa.py`, `download_avqa_videos.py` |
| Training | `train_omni.py`, `train_omni.sbatch` |
| Original inference demos | `infer_omni.py`, `infer_omni.sbatch` |
| Output review/rendering | `package_inference.py`, `render_inference_demo.py`, `transcribe_inference.py` |
| Architecture video | `render_architecture_video.py` |

The original SLS batch scripts and `sls_python.sh` depend on that installation's
paths. Use `evaluation/orcd.sbatch` for the new evaluation workflow on ORCD.
