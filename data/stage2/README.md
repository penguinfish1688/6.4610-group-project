# Stage II conversation data — unassigned

This is a central project task because synchronized real-time audiovisual
interaction data is scarce.

1. Start with one training video and a question whose answer requires vision.
2. Record when the necessary visual evidence becomes available.
3. Assemble user questions and assistant answers with timestamps. Preserve the
   source video's environment audio where it matters for audiovisual QA.
4. Use TTS for missing speech and align the resulting audio to the timeline.
5. Concatenate QA interactions while recording each clip's timestamp mapping.
6. Add interruption, overlap, silence, and response-timing variants.
7. Mix ordinary AVQA training examples and audio-only D2 replay data.
8. Convert the records to the extended trainer's tensors once that interface is
   implemented. Publish a versioned Hugging Face dataset for the team.

The [example](../examples/stage2.jsonl) is one text-only QA conversation before
TTS. The first smoke-test dataset should remain small enough to inspect manually.
Do not use held-out AVQA questions or their source videos for training.
