# D2-Omni demos

The [architecture walkthrough](architecture/README.md) is a narrated, animated
3:33 explanation of the encoders, streaming quadruplets, and training layout.

The [checkpoint-9,000 recordings](checkpoint-9000/README.md) contain three shareable
MP4s: two AVQA examples with real video and audio, and one InstructS2S example
with the encoded training placeholder. Each includes input audio and generated
speech, synchronized waveforms, and streaming text.

## Real-time audiovisual demo workstream

Extend the existing D2 browser app with timestamped webcam frames alongside
microphone audio. First verify the audio-only app on ORCD, then integrate the
vision lane after the single-video QA smoke test succeeds.

The demo needs bounded frame queues, a shared monotonic clock, explicit handling
of late/missing frames, audio playback feedback, and interruption support.
Measure both visual processing delay and time to first audible response.

Deliver a short video showing a question that depends on the camera scene and
an interruption while the assistant is speaking. This directory currently holds
the live-demo plan alongside the recorded inference videos; no audiovisual app
has been implemented yet.
