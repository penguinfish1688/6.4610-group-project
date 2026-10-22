"""Create a local review page and short speech clips from completed rollouts."""

import argparse
import html
import json
from pathlib import Path
import wave


def package(root):
    asr_path = root / "asr.json"
    asr = (
        {row["name"]: row for row in json.loads(asr_path.read_text())["examples"]}
        if asr_path.exists()
        else {}
    )
    cards = []
    for path in sorted(root.glob("*/result.json")):
        result = json.loads(path.read_text())
        folder, name = path.parent, result["name"]
        active = [e["sequence"] for e in result["events"] if e["metadata"]["emit_audio"]]
        speech = "output.wav"
        if active:
            with wave.open(str(folder / "output.wav"), "rb") as source:
                rate, params = source.getframerate(), source.getparams()
                start = max(0, round((active[0] * 0.08 - 0.16) * rate))
                end = min(source.getnframes(), round(((active[-1] + 1) * 0.08 + 0.24) * rate))
                source.setpos(start)
                data = source.readframes(end - start)
            with wave.open(str(folder / "response.wav"), "wb") as target:
                target.setparams(params)
                target.writeframes(data)
            speech = "response.wav"
        task = (
            "AVQA with real video and audio"
            if result["real_video"]
            else "InstructS2S with placeholder image"
        )
        video = (
            f'<p>Source video</p><video controls preload="metadata" width="480" src="{name}/video.mp4"></video>'
            if result["real_video"]
            else ""
        )
        recognition = asr.get(name, {})
        asr_text = (
            f"<p><b>Speech recognition:</b> {html.escape(recognition['output_asr'])}</p>"
            if recognition
            else ""
        )
        input_asr = (
            f"<p><b>Input speech recognition:</b> {html.escape(recognition['input_asr'])}</p>"
            if recognition
            else ""
        )
        cards.append(
            f"<article><h2>{html.escape(name)}: {task}</h2>"
            f"<p><b>Generated:</b> {html.escape(result['generated_text'])}</p>"
            f"<p><b>Reference:</b> {html.escape(result['reference_text'])}</p>"
            f"<p>Generated speech (silence before and after the response trimmed)</p>"
            f'<audio controls preload="metadata" src="{name}/{speech}"></audio>{asr_text}'
            f"<details><summary>Input audio, video, timing, and training prompt</summary>"
            f'{video}<p>Complete input audio</p><audio controls src="{name}/input.wav"></audio>{input_asr}'
            f'<p>Complete generated audio timeline</p><audio controls src="{name}/output.wav"></audio>'
            f"<p>Response starts: {html.escape(str(result['response_starts_seconds']))} seconds.</p>"
            f"<pre>{html.escape(result['prompt'])}</pre>"
            f'<p><a href="{name}/result.json">Frame events and provenance</a></p></details></article>'
        )
    if not cards:
        raise FileNotFoundError("No completed inference examples")
    page = (
        '<!doctype html><html><meta charset="utf-8"><title>D2-Omni inference</title>'
        "<style>body{font:18px system-ui;max-width:960px;margin:32px auto;padding:16px;line-height:1.5}"
        "article{border-top:1px solid #ccc;padding:20px 0}pre{white-space:pre-wrap}"
        "audio,video{max-width:100%}summary{cursor:pointer}</style>"
        "<h1>D2-Omni: checkpoint 1,000</h1><p>Free-running development examples, using each "
        "task's exact training prompt. AVQA receives real video and audio; InstructS2S receives "
        "audio and an encoded placeholder image. Text, speech, and self-audio feedback are generated.</p>"
        "<p>These are qualitative examples. Speech recognition is an independent automatic spot check.</p>"
        + "".join(cards)
        + "</html>"
    )
    (root / "index.html").write_text(page)
    print(root / "index.html")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("runs/inference-step1000"))
    package(parser.parse_args().run)
