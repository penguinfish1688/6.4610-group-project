# D2-Omni checkpoint 9,000

Three recorded development examples with the exact task prompts used in
training. The MP4s show synchronized video or the actual placeholder image,
input audio, generated speech, labeled waveforms, and streaming generated text.

| Video | Duration | Result |
| --- | --- | --- |
| [AVQA 1](avqa-1.mp4) | 39.52 s | `C. hail`, matching the reference. |
| [AVQA 2](avqa-2.mp4) | 38.00 s | `C. street`; reference: `A. On the road`. |
| [InstructS2S](instructs2s-1.mp4) | 35.92 s | Full three-tip answer, including all generated speech. |

Download an MP4 from its GitHub file page, or clone the repository and open
[index.html](index.html) locally to play all three. Each MP4's default soundtrack
mixes the original input audio and model speech. Players supporting audio-track
selection also offer input-only and model-speech-only tracks.

AVQA shows its full source video and holds its last frame while the question and
answer audio continue. InstructS2S shows the same fixed random placeholder image
encoded for training and inference. Original timing is preserved throughout;
the video continues after the last text token so the full speech can finish.

InstructS2S generates 12.64 seconds of active speech. Independent speech
recognition confirms the complete answer, including the diet, exercise, and
sleep tips. These three examples are qualitative development checks, not an
official test-set score. AVQA is correct on one of the two examples.

[Export verification](verification.json) records complete decoding, matching
durations, and audio comparisons against the original input and output WAVs.
All three exports have no measured speech timing offset. The default audio mix
uses a uniform gain to leave headroom; waveform lanes are individually scaled
for visibility.

The renderer is [scripts/render_inference_demo.py](../../scripts/render_inference_demo.py).
It uses FFmpeg, DejaVu Sans fonts, NumPy, Pillow, and SciPy. With completed local
inference results and an exported training placeholder image, regenerate using:

```bash
python -m scripts.render_inference_demo \
  --run runs/inference-step9000 \
  --placeholder runs/inference-step9000/demo/assets/placeholder.png
```

The generated run directory contains the original WAVs, frame events, exact
prompts, reference answers, and local checkpoint provenance. Those larger run
artifacts and model weights remain outside Git.
