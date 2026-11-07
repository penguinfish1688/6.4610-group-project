# Video with scene audio muted

Question: does the fine-tuned model answer differently when scene audio is
removed but the scene's video and the spoken question/options remain?

Use `--conditions scene_audio_muted --prepared-dataset ...`. This is evaluation
of the existing checkpoint, not a separately trained video-only model. The
vision tower alone cannot produce an answer; the Thinker and speech pipeline
remain active. Muting every audio input would also erase the spoken question.

## Processing

The prepared segment's `end_ms` marks the end of its source scene audio. Replace
the environment mel values from timeline zero through that timestamp, plus two
10-ms hops of STFT boundary context, with Qwen's zero-energy normalized floor
(-1.5). The guard may suppress up to approximately 20 ms after the scene boundary.
Retain all later mel values, the full video/DeepStack timeline, prompts, seed,
and reference. Generated self-audio feedback is unchanged. Inputs are cloned;
the unmodified correct-video condition is preserved.

These cached features retain the original running-peak normalization in later
question frames. Thus this is a feature-level removal of scene content, not an
exact waveform-muting experiment or proof that no scene-derived statistic
remains. A strict waveform protocol needs raw scene/question audio and frontend
recomputation. Compare paired accuracy with `correct`; improvements from training
without scene audio would require a separate training experiment.

Default conditions remain `correct padding mismatched`; this condition must be
requested explicitly. It requires the prepared-dataset adapter and does not
accept waveform-only demo inputs without scene timing. Original D2 remains the
audio-only baseline, rather than a meaningful scene-muted vision model.

## Do not alter a checkout while evaluations are running

The resume identity fingerprints code. Pulling updated code into an active run's
checkout may also affect modules loaded later. Use a separate worktree and reuse
the existing environment, weights and downloaded data. This does not duplicate
the 75-GB model bundle.

On ORCD, in the original checkout:

```bash
cd ~/orcd/scratch/6.4610-group-project
git fetch origin
git worktree add --detach ../6.4610-scene-audio origin/feature/avqa-evaluation
cd ../6.4610-scene-audio
export D2_OMNI_ENV="$HOME/orcd/scratch/6.4610-group-project/.venv"
module load miniforge
conda activate "$D2_OMNI_ENV"
mkdir -p logs

python -m pytest -q tests/test_evaluation.py tests/test_prepared_evaluation.py

# GPU smoke check on the first two development examples.
sbatch scripts/evaluation/orcd.sbatch \
  --bundle ../6.4610-group-project/weight/base_1030 \
  --prepared-dataset ../6.4610-group-project/data/processed/step9000-release \
  --limit 2 --conditions scene_audio_muted \
  --output runs/probe2-scene-muted-v1

# After inspecting the smoke result, evaluate the full development set.
sbatch --time=06:00:00 scripts/evaluation/orcd.sbatch \
  --bundle ../6.4610-group-project/weight/base_1030 \
  --prepared-dataset ../6.4610-group-project/data/processed/step9000-release \
  --limit 128 --conditions scene_audio_muted \
  --output runs/probe128-scene-muted-v1
```

Run only one job against each output directory. Keep this worktree and command
unchanged for resume. The ORCD script accepts `D2_OMNI_ENV` to reuse an environment
outside the checkout. Existing running/queued jobs in the original checkout
retain their old code; no cancellation or resubmission is required.

Results are written under this new worktree's `runs/`, with the same case IDs as
the existing experiments. Compare against the full model's `correct` condition.
GPU/RAM needs are unchanged; elapsed time is expected to be similar but is not
measured for this condition. Existing per-user memory limits still apply.

## Validation

13 focused evaluation/adapter tests passed; Ruff and shell syntax checks passed.
Tests check preservation of question-region features, video and references,
input isolation, and rejection of unsupported inputs. The new condition still
needs its first GPU smoke run on ORCD.
