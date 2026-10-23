"""Render synchronized, shareable videos from completed inference timelines.

The default audio track mixes the actual input and generated WAVs at their
original timestamps. Two additional selectable tracks contain each separately.
AVQA plays its source video once and holds its last frame while audio continues.
InstructS2S displays the exported training placeholder, without substituting art.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
from pathlib import Path
import subprocess
import wave

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.signal import resample_poly


WIDTH, HEIGHT, FPS = 1280, 720, 25
VIDEO = (40, 110, 704, 396)
WAVE_X, WAVE_WIDTH, WAVE_HEIGHT = 236, 1004, 52
BACKGROUND = "#0a1120"
PANEL = "#111d30"
WHITE, MUTED = "#f1f5fb", "#9aacbf"
INPUT, OUTPUT = "#52d8cc", "#ffc67a"
FONT_ROOT = Path("/usr/share/fonts/truetype/dejavu")


def font(size, bold=False):
    return ImageFont.truetype(
        str(FONT_ROOT / ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")), size
    )


def wrapped(text, face, width):
    lines = []
    for paragraph in text.splitlines():
        line = ""
        for word in paragraph.split():
            candidate = f"{line} {word}".strip()
            if line and face.getlength(candidate) > width:
                lines.append(line)
                line = word
            else:
                line = candidate
        lines.append(line)
    return lines


def paragraph(draw, xy, text, size=19, width=412, color=WHITE, spacing=25):
    x, y = xy
    for line in wrapped(text, font(size), width):
        draw.text((x, y), line, font=font(size), fill=color)
        y += spacing
    return y


def read_wav(path, rate=48000):
    with wave.open(str(path), "rb") as stream:
        if stream.getnchannels() != 1 or stream.getsampwidth() != 2:
            raise ValueError(f"Expected mono PCM16: {path}")
        original_rate = stream.getframerate()
        samples = np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2")
    samples = samples.astype(np.float64) / 32768
    common = math.gcd(rate, original_rate)
    return resample_poly(samples, rate // common, original_rate // common)


def waveform(draw, samples, y, color):
    draw.rounded_rectangle(
        (WAVE_X, y, WAVE_X + WAVE_WIDTH, y + WAVE_HEIGHT), radius=8, fill=PANEL
    )
    center = y + WAVE_HEIGHT // 2
    draw.line((WAVE_X, center, WAVE_X + WAVE_WIDTH, center), fill="#283a50")
    peak = max(float(np.max(np.abs(samples))), 1e-6)
    for x, part in enumerate(np.array_split(samples, WAVE_WIDTH)):
        low = int(center - max(0, float(part.max())) / peak * 22)
        high = int(center - min(0, float(part.min())) / peak * 22)
        draw.line((WAVE_X + x, low, WAVE_X + x, high), fill=color)


def ass_time(seconds):
    centiseconds = round(seconds * 100)
    hours, centiseconds = divmod(centiseconds, 360000)
    minutes, centiseconds = divmod(centiseconds, 6000)
    seconds, centiseconds = divmod(centiseconds, 100)
    return f"{hours}:{minutes:02}:{seconds:02}.{centiseconds:02}"


def write_captions(path, result):
    top = 374 if result["real_video"] else 294
    size = 21
    while size > 14 and len(wrapped(result["generated_text"], font(size), 412)) * size * 1.25 > 490 - top:
        size -= 1
    header = """[Script Info]
ScriptType: v4.00+
PlayResX: 1280
PlayResY: 720
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Response,DejaVu Sans,21,&H00F1F5FB,&H00F1F5FB,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,0,0,7,798,60,374,1
Style: Waiting,DejaVu Sans,18,&H00BFAC9A,&H00BFAC9A,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,0,0,7,798,60,374,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    text_events = [e for e in result["events"] if e.get("text_delta")]
    end = result["audio_seconds"]
    first = text_events[0]["sequence"] * 0.08 if text_events else end
    header = header.replace("Response,DejaVu Sans,21,", f"Response,DejaVu Sans,{size},")
    lines = [header.replace(",374,1", f",{top},1")]
    lines.append(f"Dialogue: 0,0:00:00.00,{ass_time(first)},Waiting,,0,0,0,,Listening ...\n")
    accumulated = ""
    for i, event in enumerate(text_events):
        accumulated += event["text_delta"]
        start = event["sequence"] * 0.08
        stop = text_events[i + 1]["sequence"] * 0.08 if i + 1 < len(text_events) else end
        text = "\\N".join(wrapped(accumulated.strip(), font(size), 412))
        text = text.replace("{", "(").replace("}", ")")
        lines.append(
            f"Dialogue: 0,{ass_time(start)},{ass_time(stop)},Response,,0,0,0,,{text}\n"
        )
    path.write_text("".join(lines))


def run_command(command, log):
    with log.open("w") as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"Command failed; see {log}:\n{log.read_text()[-3500:]}")


def render_case(run, result_path, output, placeholder, asr):
    result = json.loads(result_path.read_text())
    name, duration = result["name"], result["audio_seconds"]
    folder = result_path.parent
    assets = output / "assets" / name
    assets.mkdir(parents=True, exist_ok=True)
    user_audio = read_wav(folder / "input.wav")
    model_audio = read_wav(folder / "output.wav")
    length = round(duration * 48000)
    if len(user_audio) != length or len(model_audio) != length:
        raise ValueError("Input/output audio must cover the same unshifted timeline")
    mixed = user_audio + model_audio
    gain = min(1.0, 0.90 / max(float(np.max(np.abs(mixed))), 1e-6))
    mixed *= gain
    mixed_path = assets / "mixed.wav"
    with wave.open(str(mixed_path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(48000)
        stream.writeframes((mixed * 32767).round().astype("<i2").tobytes())

    canvas = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    draw.text((40, 21), "D2-Omni", font=font(34, True), fill=WHITE)
    draw.text((42, 67), "Streaming audio-visual conversation", font=font(17), fill=MUTED)
    task = "AVQA / VIDEO + AUDIO" if result["real_video"] else "INSTRUCTS2S / PLACEHOLDER IMAGE"
    draw.text((776, 29), task, font=font(19, True), fill=WHITE)
    draw.text(
        (776, 65), f"Checkpoint {result['checkpoint_update']:,}   /   {name}",
        font=font(16), fill=MUTED,
    )
    draw.rounded_rectangle((776, 110, 1240, 506), radius=14, fill=PANEL)
    draw.text((798, 130), "USER QUESTION", font=font(14, True), fill=INPUT)
    if result["real_video"]:
        annotation = result["annotation"]
        y = paragraph(draw, (798, 155), annotation["question_text"])
        for index, choice in enumerate(annotation["multi_choice"]):
            y = paragraph(
                draw, (798, y + 3), f"{chr(65 + index)}. {choice}",
                size=16, color=MUTED, spacing=22,
            )
        if y > 338:
            raise ValueError("Question/choices exceed the demo panel")
    else:
        transcript = asr.get(name, {}).get("input_asr", "Listen to the input audio.")
        paragraph(draw, (798, 157), transcript.capitalize(), size=19)
    response_label_y = 351 if result["real_video"] else 271
    draw.line((798, response_label_y - 10, 1218, response_label_y - 10), fill="#2a3c52")
    draw.text((798, response_label_y), "MODEL RESPONSE", font=font(14, True), fill=OUTPUT)
    draw.text((40, 538), "INPUT AUDIO", font=font(16, True), fill=INPUT)
    draw.text((40, 562), "Video / user speech" if result["real_video"] else "User speech", font=font(13), fill=MUTED)
    draw.text((40, 611), "MODEL SPEECH", font=font(16, True), fill=OUTPUT)
    draw.text((40, 635), "Generated audio", font=font(13), fill=MUTED)
    waveform(draw, user_audio, 535, INPUT)
    waveform(draw, model_audio, 608, OUTPUT)
    draw.text((40, 686), "Input audio + model speech / original timing", font=font(13), fill=MUTED)
    draw.text((904, 686), f"/ {duration:.2f} s", font=font(13), fill=MUTED)
    background = assets / "background.png"
    canvas.save(background)
    captions = assets / "captions.ass"
    write_captions(captions, result)

    if result["real_video"]:
        video = Path(result["video"]).resolve()
        visual_input = ["-i", str(video)]
        video_filter = (
            f"[0:v]setpts=PTS-STARTPTS,fps={FPS},scale=704:396:force_original_aspect_ratio=decrease,"
            "pad=704:396:(ow-iw)/2:(oh-ih)/2:color=0x050a12,setsar=1,"
            f"tpad=stop_mode=clone:stop_duration={duration}[visual]"
        )
        source_seconds = float(json.loads(subprocess.check_output([
            "ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(video)
        ], text=True))["format"]["duration"])
    else:
        if not placeholder.is_file():
            raise FileNotFoundError("Export the exact training placeholder before rendering")
        visual_input = ["-loop", "1", "-framerate", str(FPS), "-i", str(placeholder)]
        video_filter = (
            "[0:v]scale=528:396:flags=neighbor,pad=704:396:(ow-iw)/2:(oh-ih)/2:"
            "color=0x050a12,setsar=1[visual]"
        )
        source_seconds = None
    overlay = f"[1:v][visual]overlay=40:110:shortest=1[scene]"
    # drawtext's x expression is evaluated per video frame, so both playheads
    # share exactly the same t as the video and the unshifted mixed soundtrack.
    cursor = ";".join(
        f"drawtext=fontfile={FONT_ROOT / 'DejaVuSans.ttf'}:text='|':fontsize=54:"
        f"fontcolor=white:x='{WAVE_X}-8+min(t/{duration}\\,1)*{WAVE_WIDTH}':y={y - 8}"
        for y in (535, 608)
    ).replace(";", ",")
    label = "LAST VIDEO FRAME / AUDIO CONTINUES" if result["real_video"] else "ENCODED PLACEHOLDER IMAGE"
    label_enable = f":enable='gte(t,{source_seconds})'" if source_seconds else ""
    filter_graph = ";".join([
        video_filter,
        overlay,
        f"[scene]{cursor},drawtext=fontfile={FONT_ROOT / 'DejaVuSans.ttf'}:"
        f"text='{label}':fontsize=13:fontcolor=white:box=1:boxcolor=black@0.7:"
        f"boxborderw=7:x=58:y=477{label_enable},"
        f"drawtext=fontfile={FONT_ROOT / 'DejaVuSans.ttf'}:text='%{{pts\\:hms}}':"
        f"fontsize=13:fontcolor=0x9aacbf:x=792:y=686,ass=filename='{captions}'[demo]",
    ])
    destination = output / f"{name}.mp4"
    command = [
        "ffmpeg", "-hide_banner", "-y", "-threads", "4", *visual_input,
        "-loop", "1", "-framerate", str(FPS), "-i", str(background),
        "-i", str(mixed_path), "-i", str(folder / "input.wav"), "-i", str(folder / "output.wav"),
        "-filter_complex_threads", "2", "-filter_complex", filter_graph,
        "-map", "[demo]", "-map", "2:a", "-map", "3:a", "-map", "4:a",
        "-t", str(duration), "-r", str(FPS), "-c:v", "libx264", "-preset", "fast",
        "-crf", "19", "-pix_fmt", "yuv420p", "-threads", "4",
        "-c:a", "aac", "-b:a", "160k", "-ar", "48000",
        "-metadata:s:a:0", "title=Input + model speech", "-metadata:s:a:1", "title=Input only",
        "-metadata:s:a:2", "title=Model speech only",
        "-metadata:s:a:0", "handler_name=Input + model speech",
        "-metadata:s:a:1", "handler_name=Input only",
        "-metadata:s:a:2", "handler_name=Model speech only",
        "-disposition:a:0", "default",
        "-disposition:a:1", "0", "-disposition:a:2", "0", "-movflags", "+faststart",
        str(destination),
    ]
    (assets / "ffmpeg-command.json").write_text(json.dumps(command, indent=2) + "\n")
    run_command(command, assets / "render.log")
    streams = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(destination)
    ], text=True))
    if abs(float(streams["format"]["duration"]) - duration) > 0.08:
        raise ValueError("Rendered demo duration differs from the inference timeline")
    videos = [s for s in streams["streams"] if s["codec_type"] == "video"]
    audios = [s for s in streams["streams"] if s["codec_type"] == "audio"]
    assert len(videos) == 1 and (videos[0]["width"], videos[0]["height"]) == (WIDTH, HEIGHT)
    assert len(audios) == 3 and audios[0]["disposition"]["default"] == 1
    receipt = dict(
        name=name, file=destination.name, checkpoint=result["checkpoint_update"],
        duration_seconds=duration, source_video=str(video) if result["real_video"] else str(placeholder),
        source_video_seconds=source_seconds, video_tail="hold last frame" if result["real_video"] else "static placeholder",
        default_audio="input + generated speech at original timestamps", audio_mix_gain=gain,
        selectable_audio_tracks=["Input + model speech", "Input only", "Model speech only"],
        waveform_display="Each lane is independently peak-scaled for visibility; timing is unchanged",
        generated_text=result["generated_text"], reference_text=result["reference_text"],
        input_sha256=hashlib.sha256((folder / "input.wav").read_bytes()).hexdigest(),
        generated_audio_sha256=hashlib.sha256((folder / "output.wav").read_bytes()).hexdigest(),
        ffprobe=streams,
    )
    (assets / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({k: receipt[k] for k in ("name", "file", "checkpoint", "duration_seconds")}), flush=True)
    return receipt


def main(args):
    output = args.run / "demo"
    output.mkdir(parents=True, exist_ok=True)
    asr_path = args.run / "asr.json"
    asr = {r["name"]: r for r in json.loads(asr_path.read_text())["examples"]} if asr_path.exists() else {}
    paths = sorted(args.run.glob("*/result.json"))
    if len(paths) != 3:
        raise ValueError("The demo expects two completed AVQA cases and one InstructS2S case")
    results = [render_case(args.run, p, output, args.placeholder, asr) for p in paths]
    (output / "manifest.json").write_text(json.dumps(results, indent=2) + "\n")
    cards = [
        f'<article><h2>{html.escape(r["name"])}</h2><video controls preload="metadata" '
        f'src="{r["file"]}"></video><p><a download href="{r["file"]}">Download MP4</a></p></article>'
        for r in results
    ]
    (output / "index.html").write_text(
        '<!doctype html><html><meta charset="utf-8"><title>D2-Omni demo</title>'
        '<style>body{background:#0a1120;color:#f1f5fb;font:17px system-ui;max-width:1100px;'
        'margin:32px auto;padding:20px}video{width:100%}a{color:#52d8cc}article{margin:36px 0}</style>'
        f'<h1>D2-Omni / checkpoint {results[0]["checkpoint"]:,}</h1>'
        '<p>Three unedited development rollouts. Each video includes the input audio and generated '
        'speech with synchronized waveforms and streaming text. AVQA shows the source video; '
        'InstructS2S shows the actual placeholder image.</p>' + "".join(cards) + '</html>'
    )
    print(output / "index.html", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--placeholder", type=Path, required=True)
    main(parser.parse_args())
