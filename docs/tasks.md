# Task board

Add your name and a short plan when claiming work. Update the status and link
artifacts in [updates.md](updates.md); leave unclaimed tasks unassigned.

| Task | Owner | First deliverable | Status |
| --- | --- | --- | --- |
| Stage I vision encoder and data | CK | Native encoder profile, video subset, compression/distillation plan | Planned |
| Stage II dataset | Unassigned | One aligned video-QA sample, then a versioned Hugging Face dataset | Planned |
| Stage II model integration/training | Unassigned | Overfit one streaming video-QA sample | Planned |
| Benchmarking | Unassigned | AVQA split/protocol, baselines, latency measurement scripts | Planned |
| Real-time webcam demo | Unassigned | Timestamped camera/microphone input and speech output | Planned |
| Literature review | Unassigned | Related work with sources and comparable settings | Planned |
| Writing | Unassigned | Methods/results outline backed by actual experiments | Planned |
| Presentation | Unassigned | Slides and a short demonstration video | Planned |
| GRPO for QA | Unassigned | Proposal after the minimum viable result | Stretch |

## Milestones

1. Reproduce the audio-only D2 demo on an ORCD GPU with enough VRAM.
2. Measure native vision encoding cost and settle the first feature interface.
3. Build a small synchronized dataset and overfit one video-QA example.
4. Train a larger audiovisual model and evaluate held-out AVQA plus audio retention.
5. Record the webcam demo, measure latency, and prepare the report/presentation.

No deadlines or assignments beyond CK's Stage I ownership have been assumed.
