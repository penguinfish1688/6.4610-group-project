# Real-time audiovisual demo workstream

Extend the existing D2 browser app with timestamped webcam frames alongside
microphone audio. First verify the audio-only app on ORCD, then integrate the
vision lane after the single-video QA smoke test succeeds.

The demo needs bounded frame queues, a shared monotonic clock, explicit handling
of late/missing frames, audio playback feedback, and interruption support.
Measure both visual processing delay and time to first audible response.

Deliver a short video showing a question that depends on the camera scene and
an interruption while the assistant is speaking. This directory currently holds
the demo plan; no audiovisual app has been implemented yet.
