# Stage I data — CK

Prepare video-only training manifests and frame sampling/teacher feature tools.
Candidate sources: AVQA training videos, MSR-VTT, ActivityNet Captions.

First deliverable: a small reproducible clip subset, timestamped frames, and a
native-encoder profile. Then add feature caching and distillation batches after
the teacher/student representation and token budget are agreed.

Use the [draft clip record](../examples/stage1.jsonl). Record source provenance,
split, preprocessing, and teacher revision next to cached features. Data acquisition
and feature extraction have not been implemented yet.
