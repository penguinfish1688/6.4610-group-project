# Evaluation workstream

First implement a reproducible held-out AVQA runner. Record the exact dataset
variant/version, video-level split, prompt, answer normalization, metric, and
checkpoint revision. Do not train on test videos, including during Stage I.

Measure:

- **Audiovisual QA:** AVQA accuracy and modality ablations (audio-only,
  vision-only, audiovisual, mismatched/removed video).
- **Speech retention:** audio-only QA and existing D2 benchmark comparisons.
- **Full-duplex behavior:** turn-taking, interruptions, overlap, and silence.
- **Latency:** preprocessing/encoder compute, time to first text/audio, turn-end
  and interruption latency, real-time factor, memory, and p50/p95 distributions.
- **Optional video understanding:** Video-MME if time permits.

Use original Qwen3-Omni and MiniCPM-o 4.5 as proposed baselines when hardware,
modalities, prompts, and protocols are comparable. Keep an original D2 baseline
for audio retention. Record frame rate, resolution, visual tokens per second,
context length, warmup, hardware, and software versions with every latency result.

D2's existing benchmark documentation is under `model/backbone/benchmarks/`.
Initialize its pinned benchmark submodules from the repository root when needed.
The D2-Omni AVQA/latency runners are not implemented yet.
