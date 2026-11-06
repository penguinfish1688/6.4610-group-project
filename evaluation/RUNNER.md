# Paired AVQA evaluation on ORCD

The runner evaluates prepared AVQA inputs with the released step-9,000 model.
The bundled two AVQA cases are debugging examples, not a benchmark. InstructS2S
is excluded. No training or live visual encoding is performed.

The new [step-9000 development adapter](STEP9000.md) can evaluate the 128
released AVQA probe clips directly from their cached environment mel and visual
features, and compare the original D2 weights with the fine-tuned model. This
uses `--prepared-dataset`; the commands and processing differences are documented
in that guide.

Implementation files are grouped under `scripts/evaluation/`; see the
[scripts directory guide](../scripts/README.md). The package entry point is
`python -m scripts.evaluation`. The earlier top-level `scripts.evaluate_omni`
command has been replaced. Because resume checks fingerprint code, use a new
output directory if you already ran the previous layout.

## Preview without a GPU

Run from the repository root in the Python environment used for reproduction:

```bash
python -m scripts.evaluation --dry-run
```

The defaults select both AVQA cases and three conditions (six rollouts):

- `correct`: original visual features, DeepStack, and mask.
- `padding`: all-false visual mask and zeroed feature tensors. The model uses its
  learned padding embedding and supplies no DeepStack features.
- `mismatched`: another selected video's features and DeepStack at the same tick.
  Only ticks valid in both videos are used; unmatched tails are padded, not looped.
  Donors are deterministic and exclude identical `source_video_id` values when
  available, otherwise identical video paths. Add source IDs to larger manifests.

All conditions retain the same four-position clock, audio, native task prompt,
seed, and full response tail. Every rollout starts a fresh session. References
are used for scoring only. An all-padding input is a distribution shift, so
interpret it together with mismatched video, not as equivalent to original D2.
Mismatched coverage can be lower for unequal-duration videos; inspect visual-token
counts, and use comparable-duration donors for a larger controlled study.

## Run on ORCD

Use the same `.venv` and downloaded release as for successful reproduction.
Submit from the checkout root (create logs before submission):

```bash
mkdir -p logs
sbatch scripts/evaluation/orcd.sbatch --output runs/avqa-eval-demo-v1
```

The batch script requests one L40S, 180 GB host RAM, eight CPUs and two hours on
`mit_normal_gpu`. Adjust allocation/account settings to your ORCD access and
increase the time limit for larger sets. It uses ORCD Miniforge and your local
`.venv`, not CK's SLS wrapper. A submitted batch job survives SSH disconnects.

Inside an existing GPU allocation, run directly instead:

```bash
python -u -m scripts.evaluation --output runs/avqa-eval-demo-v1
```

Use `--conditions correct padding` to start with two conditions, or
`--cases avqa-1 avqa-2` to select explicit names. Mismatched video requires at
least two distinct selected videos. No CLI option changes the seed; it comes
from the release configuration, matching reproduction.

Rerunning the exact command resumes completed cases. Settings, prepared manifest,
checkpoint manifest, runtime package versions and relevant code hashes must
match `evaluation-run.json`. Use a new output directory after changes. An
interrupted case without `result.json` is rerun; completed cases retain their
audio and result. The model is loaded once per job, even on resume.

## Results

- `evaluation.html`: answer comparison and generated speech, updated after each case.
- `evaluation.csv`: one row per completed case/condition.
- `evaluation.json`: completion counts, accuracy, invalid-answer counts, paired
  accuracy differences and runtime. Partial runs are explicitly labeled.
- `<case>--<condition>/`: input/output WAVs, text, result and generation events.
- `results.json`, `index.html`: existing inference outputs (the latter is a generic
  demo page; use `evaluation.html` for labeled ablation comparisons).

Scoring accepts a leading A/B/C/D choice, including `Answer: C` or `The answer is C`.
It does not match letters embedded in arbitrary prose or guess from option wording.
Unparseable answers are incorrect and separately counted. Inspect raw text to
audit the parser. Active speech duration is not an automatic completion score.
Timings exclude model loading, cached feature preparation and live vision encoding.
The runner scores generated text; transcribed speech accuracy needs a separate ASR pass.

## Add held-out data later

Use `--manifest PATH --data-root ROOT`. The manifest uses the exact existing
`weight/base_1030/examples/manifest.json` schema, including input SHA256, prompt
IDs, reference, visual-token count, annotation and timing fields. Input/video
paths resolve inside ROOT. The original model bundle remains specified by
`--bundle`. The release loader verifies the input checksum and native task prompt.

Prepared safetensors contain `audio` (16-kHz waveform) and `vision_features`,
`vision_mask`, `vision_deepstack` aligned to the entire 80-ms timeline including
the response tail. Add `source_video_id` to manifest rows to distinguish original
video identities for donor selection. This does not replace a train/development
overlap audit: verify source-video disjointness against CK's training inventory
before reporting validation performance. `--limit N` takes manifest order, so
create a fixed representative manifest rather than treating the first N as random.
