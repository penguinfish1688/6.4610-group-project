"""Build the narrated D2-Omni architecture film from the reviewed storyboard.

Narration uses a local Piper voice; diagrams are editable Pillow drawing code.
All large intermediates stay in .cache. No GPU or training data is required.
"""

from __future__ import annotations

import argparse
import ast
from functools import lru_cache
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import subprocess
import wave

import numpy as np
from PIL import Image, ImageDraw, ImageFont


W, H, FPS = 1920, 1080, 24
BG, PANEL, BORDER = "#0a1120", "#142137", "#2b3b54"
WHITE, MUTED = "#f0f5fc", "#9aadc4"
U, A, V, T = "#53dccc", "#ffa96b", "#bba0ff", "#ffda7d"
GREEN, RED = "#77dfa4", "#ff838c"
COLORS = [U, A, V, T]
FONT_DIR = Path("/usr/share/fonts/truetype/lato")


@lru_cache(maxsize=80)
def font(size, bold=False, mono=False):
    if mono:
        path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf")
    else:
        path = FONT_DIR / ("Lato-Bold.ttf" if bold else "Lato-Regular.ttf")
    return ImageFont.truetype(str(path), size)


def wrap(text, size, width, bold=False, mono=False):
    face = font(size, bold, mono)
    lines = []
    for paragraph in text.split("\n"):
        line = ""
        for word in paragraph.split():
            candidate = (line + " " + word).strip()
            if line and face.getlength(candidate) > width:
                lines.append(line)
                line = word
            else:
                line = candidate
        lines.append(line)
    return lines


def text(draw, x, y, value, size=26, color=WHITE, bold=False, mono=False, width=None):
    lines = wrap(value, size, width, bold, mono) if width else value.split("\n")
    for line in lines:
        draw.text((x, y), line, font=font(size, bold, mono), fill=color, anchor="lt")
        y += int(size * 1.30)
    return y


def center(draw, x, y, width, value, size=25, color=WHITE, bold=False, mono=False):
    offset = (width - font(size, bold, mono).getlength(value)) / 2
    text(draw, x + offset, y, value, size, color, bold, mono)


def card(draw, box, title, body="", color=BORDER, size=27, body_size=23):
    x, y, width, height = box
    while size > 20 and font(size, True).getlength(title) > width - 50:
        size -= 1
    draw.rounded_rectangle(
        (x, y, x + width, y + height), radius=18, fill=PANEL, outline=color, width=2
    )
    text(draw, x + 25, y + 23, title, size, WHITE, True, width=width - 50)
    if body:
        text(draw, x + 25, y + 65, body, body_size, MUTED, width=width - 50)


def pill(draw, box, value, color, size=25):
    x, y, width, height = box
    draw.rounded_rectangle(
        (x, y, x + width, y + height), radius=12, fill=PANEL, outline=color, width=2
    )
    center(draw, x, y + (height - size) / 2 - 1, width, value, size, color, True)


def arrow(draw, points, color=MUTED, width=4):
    draw.line(points, fill=color, width=width, joint="curve")
    x, y = points[-1]
    px, py = points[-2]
    angle = math.atan2(y - py, x - px)
    length = 13
    draw.polygon(
        [
            (x, y),
            (x - length * math.cos(angle - 0.5), y - length * math.sin(angle - 0.5)),
            (x - length * math.cos(angle + 0.5), y - length * math.sin(angle + 0.5)),
        ],
        fill=color,
    )


def packet(draw, points, progress, color, radius=8):
    segments = [math.dist(a, b) for a, b in zip(points, points[1:])]
    position = (progress % 1) * sum(segments)
    for a, b, length in zip(points, points[1:], segments):
        if position <= length:
            ratio = position / max(length, 1e-9)
            x, y = a[0] + ratio * (b[0] - a[0]), a[1] + ratio * (b[1] - a[1])
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
            return
        position -= length


def highlight(draw, box, color, width=4):
    x, y, w, h = box
    draw.rounded_rectangle(
        (x - 5, y - 5, x + w + 5, y + h + 5), radius=18, outline=color, width=width
    )


def short_wave(draw, box, phase=0, color=U):
    x, y, width, height = box
    center_y = y + height / 2
    for j in range(70):
        envelope = 0.18 + 0.82 * abs(math.sin(j * 0.12 + phase))
        amplitude = envelope * (height / 2 - 2) * (0.2 + 0.8 * abs(math.sin(j * 1.73)))
        xx = x + j * width / 70
        draw.line((xx, center_y - amplitude, xx, center_y + amplitude), fill=color, width=3)


def stamp(seconds, srt=False):
    value = round(seconds * 1000)
    hours, value = divmod(value, 3600000)
    minutes, value = divmod(value, 60000)
    seconds, millis = divmod(value, 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02}{',' if srt else '.'}{millis:03}"


def validate_sources(book):
    cfg = json.loads(Path("configs/train-sls.json").read_text())
    assert cfg["vision"] == dict(
        fps=1,
        height=96,
        width=128,
        tokens_per_image=12,
        placeholder_seed=1337,
        encoder_frozen=True,
        distillation=False,
    )
    assert cfg["max_frames"] == cfg["memory_frames"] == 1125
    assert cfg["minimum_conversation_seconds"] == 720
    assert cfg["control_offset_frames"] == 6 and cfg["talker_codec_eos_weight"] == 5
    source = ast.parse(Path("model/backbone/d2/prompts.py").read_text())
    prompts = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in source.body
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
    }
    assert book["prompts"]["conversation"] == prompts["CONVERSATION_SYSTEM_PROMPT"]
    assert book["prompts"]["avqa"] == prompts["AVQA_SYSTEM_PROMPT"]
    return dict(
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        checked_contract=dict(
            tick_ms=80,
            image_fps=1,
            visual_tokens_per_image=12,
            ticks_per_two_seconds=25,
            visual_padding_per_two_seconds=1,
            chunk_seconds=90,
            memory_seconds=90,
            chunk_ticks=1125,
            stream_positions_per_chunk=4500,
            initial_prompt_is_extra=True,
        ),
        source_files={
            p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in book["source_files"]
        },
    )


def narrate(args, book):
    import onnxruntime as ort
    from piper import PiperVoice, SynthesisConfig
    from piper.config import PiperConfig

    cache = args.cache
    cache.mkdir(parents=True, exist_ok=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    ort.disable_telemetry_events()
    voice = PiperVoice(
        session=ort.InferenceSession(
            str(args.voice), sess_options=options, providers=["CPUExecutionProvider"]
        ),
        config=PiperConfig.from_dict(json.loads(Path(str(args.voice) + ".json").read_text())),
    )
    config = SynthesisConfig(length_scale=1.0, noise_scale=0.60, noise_w_scale=0.75, volume=0.90)
    rate = voice.config.sample_rate
    sound, cues, scenes = bytearray(), [], []
    time = 0.0

    def silence(seconds):
        nonlocal time
        count = round(seconds * rate)
        sound.extend(b"\0\0" * count)
        time += count / rate

    for scene_index, scene in enumerate(book["scenes"]):
        start = time
        silence(0.4)
        for sentence_index, sentence in enumerate(scene["sentences"]):
            digest = hashlib.sha256(sentence["spoken"].encode()).hexdigest()[:12]
            path = cache / f"voice-{scene_index:02}-{sentence_index:02}-{digest}.wav"
            if not path.exists():
                with wave.open(str(path), "wb") as stream:
                    voice.synthesize_wav(sentence["spoken"], stream, syn_config=config)
            with wave.open(str(path), "rb") as stream:
                assert stream.getframerate() == rate and stream.getnchannels() == 1
                samples = stream.getnframes()
                audio = stream.readframes(samples)
            cues.append(
                dict(
                    start=time,
                    end=time + samples / rate,
                    text=sentence["caption"],
                    spoken=sentence["spoken"],
                    scene=scene_index,
                )
            )
            sound.extend(audio)
            time += samples / rate
            silence(0.13)
        silence(0.65)
        # All scene boundaries land on a video-frame boundary.
        target = math.ceil(time * FPS) / FPS
        silence(target - time)
        scenes.append(dict(id=scene["id"], title=scene["chapter"], start=start, end=time))
        print(
            json.dumps(dict(event="narrated", scene=scene["id"], seconds=round(time - start, 2))),
            flush=True,
        )
    with wave.open(str(cache / "narration.wav"), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes(sound)
    timeline = dict(
        duration=time,
        sample_rate=rate,
        scenes=scenes,
        cues=cues,
        piper_version=importlib.metadata.version("piper-tts"),
        voice=str(args.voice),
    )
    (cache / "timeline.json").write_text(json.dumps(timeline, indent=2) + "\n")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "captions.srt").write_text(
        "\n".join(
            f"{i + 1}\n{stamp(c['start'], True)} --> {stamp(c['end'], True)}\n{c['text']}\n"
            for i, c in enumerate(cues)
        )
    )
    (args.output / "chapters.json").write_text(json.dumps(scenes, indent=2) + "\n")
    transcript = [book["title"], ""]
    for scene, timing in zip(book["scenes"], scenes):
        offset = int(timing["start"])
        transcript += [
            f"{offset // 60}:{offset % 60:02} — {scene['chapter']}",
            *(sentence["caption"] for sentence in scene["sentences"]),
            "",
        ]
    (args.output / "transcript.txt").write_text("\n".join(transcript).rstrip() + "\n")
    print(
        json.dumps(dict(event="narration_complete", seconds=time, sentences=len(cues))), flush=True
    )


class Diagrams:
    def __init__(self, book):
        self.book = book
        self.backgrounds = {}
        for index, scene in enumerate(book["scenes"]):
            image = Image.new("RGB", (W, H), BG)
            draw = ImageDraw.Draw(image)
            text(draw, 80, 35, "D2-OMNI / ARCHITECTURE", 21, U, True)
            text(draw, 80, 92, scene["title"], 54, WHITE, True)
            text(draw, 82, 168, scene["subtitle"], 27, MUTED)
            text(draw, 1735, 40, f"{index + 1:02} / {len(book['scenes']):02}", 25, MUTED, mono=True)
            getattr(self, f"base_{scene['id']}")(image, draw)
            text(draw, 80, 896, "SCHEMATIC / animation slowed for explanation", 16, MUTED)
            names = [
                "Architecture",
                "Audio",
                "Vision",
                "Inference",
                "Feedback",
                "Prompts",
                "Interleaving",
                "XL memory",
                "Objectives",
            ]
            for j, name in enumerate(names):
                pill(draw, (80 + j * 197, 930, 184, 31), name, U if j == index else BORDER, 16)
            self.backgrounds[scene["id"]] = image

    def base_overview(self, image, d):
        card(d, (90, 285, 335, 115), "User / scene audio", "The current 80 ms", U)
        card(d, (90, 465, 335, 115), "Previous speech", "Delayed self feedback", A)
        card(d, (90, 690, 335, 115), "Video / placeholder", "A frame every second", V)
        card(
            d,
            (505, 325, 385, 245),
            "Causal audio encoder",
            "Shared weights\nIndependent stream states",
            U,
        )
        card(d, (505, 690, 385, 115), "Frozen vision encoder", "12 tokens per image", V)
        boxes = [(995, 290 + j * 118, 395, 90) for j in range(4)]
        for box, name, color in zip(
            boxes,
            ["1  User audio U[i]", "2  Self audio A[i]", "3  Vision V[i]", "4  Previous text"],
            COLORS,
        ):
            pill(d, box, name, color, 26)
        card(d, (1495, 380, 330, 165), "Thinker + KV cache", "Four positions per tick", U)
        card(d, (1495, 645, 330, 100), "Text / control", color=T)
        card(d, (1495, 790, 330, 90), "Talker → speech", color=A)
        for points, color in [
            ([(425, 345), (505, 345)], U),
            ([(425, 525), (505, 525)], A),
            ([(425, 750), (505, 750)], V),
            ([(890, 390), (940, 390), (940, 335), (995, 335)], U),
            ([(890, 505), (960, 505), (960, 453), (995, 453)], A),
            ([(890, 750), (945, 750), (945, 571), (995, 571)], V),
            ([(1660, 545), (1660, 645)], T),
            ([(1660, 745), (1660, 790)], A),
        ]:
            arrow(d, points, color)
        d.line((1440, 335, 1440, 689), fill=MUTED, width=3)
        for j, color in enumerate(COLORS):
            d.line((1390, 335 + j * 118, 1440, 335 + j * 118), fill=color, width=3)
        arrow(d, [(1440, 453), (1495, 453)], MUTED)
        arrow(d, [(1495, 695), (1460, 695), (1460, 689), (1390, 689)], T, 2)
        text(d, 995, 805, "z[i] becomes the next\nprevious-text input", 23, T)

    def base_audio(self, image, d):
        card(d, (90, 310, 315, 175), "User / scene PCM", "16 kHz input", U)
        card(d, (90, 615, 315, 175), "Previous self PCM", "Generated S[i−1]", A)
        card(d, (475, 310, 270, 175), "Causal log-mel", "Past + current audio", U, 26, 21)
        card(d, (475, 615, 270, 175), "Causal log-mel", "Separate self state", A, 26, 21)
        card(
            d,
            (820, 280, 565, 530),
            "D2 causal audio encoder",
            "Same parameters for both lanes",
            U,
            32,
            26,
        )
        card(d, (855, 405, 495, 130), "User stream state", "Carried across successive ticks", U)
        card(d, (855, 615, 495, 130), "Self stream state", "Carried independently", A)
        pill(d, (1470, 360, 335, 115), "U[i] = E(X[i])", U, 30)
        pill(d, (1470, 650, 335, 115), "A[i] = E(S[i−1])", A, 27)
        for y, color in [(400, U), (700, A)]:
            arrow(d, [(405, y), (475, y)], color)
            arrow(d, [(745, y), (820, y)], color)
            arrow(d, [(1385, y), (1470, y)], color)
        center(
            d,
            420,
            843,
            1170,
            "80 ms / tick  =  12.5 embeddings per second, per lane",
            29,
            WHITE,
            True,
        )

    def base_vision(self, image, d):
        card(d, (85, 295, 330, 265), "Current image", "Only after capture time", V)
        # A schematic scene avoids any dependency on external media or assets.
        d.rectangle((110, 395, 390, 530), fill="#263f58")
        d.polygon([(110, 530), (200, 420), (280, 485), (340, 430), (390, 530)], fill="#507687")
        d.ellipse((335, 405, 365, 435), fill=T)
        card(d, (495, 350, 300, 150), "Resize each image", "96 H × 128 W pixels", V)
        card(
            d,
            (885, 330, 350, 190),
            "Frozen vision encoder",
            "Native patch + merger\nNo vision distillation",
            V,
        )
        text(d, 1340, 282, "12 merged tokens / image", 29, V, True)
        for j in range(12):
            pill(d, (1345 + j % 4 * 115, 345 + j // 4 * 62, 98, 47), f"v{j}", V, 22)
        for a, b in [
            ((415, 430), (495, 430)),
            ((795, 430), (885, 430)),
            ((1235, 430), (1330, 430)),
        ]:
            arrow(d, [a, b], V)
        arrow(d, [(1060, 520), (1060, 580), (1640, 580)], V)
        text(d, 1085, 545, "Matching DeepStack → visual positions", 23, V)
        text(d, 85, 643, "VISUAL QUEUE / one position every 80 ms", 23, WHITE, True)
        for j in range(25):
            color = MUTED if j == 12 else V if j < 12 else "#78b1ff"
            label = "PAD" if j == 12 else str(j if j < 12 else j - 13)
            pill(d, (88 + j * 70, 705, 60, 67), label, color, 18 if j == 12 else 23)
        text(d, 90, 800, "Frame at 0 s: 12 tokens", 25, V)
        text(d, 932, 800, "pad", 22, MUTED)
        text(d, 1080, 800, "Frame at 1 s: 12 tokens", 25, "#78b1ff")
        center(
            d,
            100,
            852,
            1720,
            "25 ticks / 2 seconds = 24 real visual tokens + 1 visual pad",
            28,
            WHITE,
            True,
        )

    def base_inference(self, image, d):
        pill(
            d,
            (100, 255, 1720, 68),
            "STREAM START: task-specific system prompt → initial Thinker cache H[0]",
            V,
            27,
        )
        d.rounded_rectangle((115, 370, 1375, 585), radius=23, outline=U, width=2)
        text(d, 140, 385, "CALL 1 / prefill three positions", 24, U, True)
        labels = ["U[i]", "A[i]", "V[i]", "z[i−1]"]
        for j, (name, color) in enumerate(zip(labels, COLORS)):
            pill(d, (145 + j * 425, 445, 350, 100), name, color, 41)
        text(d, 1420, 385, "CALL 2 / text step", 24, T, True)
        card(d, (100, 675, 290, 110), "Existing KV", "H before tick i", BORDER, 28)
        card(
            d,
            (530, 640, 850, 180),
            "Thinker / shared causal KV cache",
            "Append U[i], A[i], V[i], then previous text z[i−1]",
            U,
            34,
            26,
        )
        card(d, (1510, 655, 310, 140), "Predict z[i]", "Text or control token", T, 33, 23)
        arrow(d, [(390, 730), (530, 730)], MUTED)
        arrow(d, [(1380, 730), (1510, 730)], T)
        arrow(d, [(720, 585), (720, 640)], U)
        arrow(d, [(1595, 545), (1595, 595), (1200, 595), (1200, 640)], T)
        center(
            d,
            100,
            850,
            1720,
            "Four ordered positions per tick; generated z[i] is the text input on tick i+1",
            27,
            WHITE,
        )

    def base_feedback(self, image, d):
        card(
            d,
            (90, 285, 345, 180),
            "Thinker",
            "Text / control token\nLayer-24 hidden state",
            U,
            33,
            25,
        )
        card(
            d,
            (525, 285, 415, 180),
            "Native response routing",
            "Text context while speaking\nHidden context while listening",
            T,
            28,
            24,
        )
        card(d, (1030, 300, 300, 150), "Talker + predictor", "Speech-code frame", A, 28, 24)
        card(d, (1430, 300, 390, 150), "Waveform renderer", "S[i] / 80 ms of audio", A, 31, 25)
        card(
            d,
            (575, 555, 470, 105),
            "Self audio encoder",
            "Generated playback, on tick i+1",
            A,
            28,
            23,
        )
        for points, color in [
            ([(435, 375), (525, 375)], U),
            ([(940, 375), (1030, 375)], T),
            ([(1330, 375), (1430, 375)], A),
            ([(1625, 450), (1625, 610), (1045, 610)], A),
            ([(575, 610), (265, 610), (265, 465)], A),
        ]:
            arrow(d, points, color)
        text(d, 1160, 547, "24 kHz audio → 16 kHz feedback", 22, MUTED)
        text(d, 90, 728, "TEXT", 24, T, True)
        text(d, 90, 810, "SPEECH", 24, A, True)
        for j in range(7):
            pill(d, (355 + j * 91, 715, 76, 47), "z", T, 22)
        arrow(d, [(1015, 739), (1775, 739)], BORDER, 2)
        text(d, 1075, 707, "Text finished; audio continues", 25, MUTED)
        short_wave(d, (355, 800, 1310, 52), 0.7, A)
        pill(d, (1685, 797, 130, 58), "EOS", A, 25)

    def base_prompts(self, image, d):
        card(d, (90, 265, 840, 370), "InstructS2S / conversation", color=U, size=33)
        card(d, (990, 265, 840, 370), "AVQA / multiple choice", color=V, size=33)
        text(d, 120, 340, '"' + self.book["prompts"]["conversation"] + '"', 28, WHITE, width=770)
        text(d, 1020, 340, '"' + self.book["prompts"]["avqa"] + '"', 28, WHITE, width=770)
        rng = np.random.default_rng(1337)
        placeholder = Image.fromarray(rng.integers(0, 256, (96, 128, 3), dtype=np.uint8))
        image.paste(placeholder.resize((128, 96), Image.Resampling.NEAREST), (125, 510))
        text(d, 280, 530, "Fixed random image → vision encoder", 26, U)
        pill(d, (1020, 515, 190, 82), "Real video", V, 26)
        text(d, 1240, 530, "Scene audio + full video", 26, V)
        text(
            d,
            90,
            671,
            "FIRST CHUNK / ordinary prompt tokens, then aligned modality positions",
            24,
            WHITE,
            True,
        )
        card(d, (90, 725, 365, 112), "SYSTEM PROMPT", "<|im_start|>system ...", V, 27, 20)
        for j in range(3):
            start = 535 + j * 425
            for k, (name, color) in enumerate(zip(["U", "A", "V", "Tprev"], COLORS)):
                pill(d, (start + k * 91, 745, 80, 65), name, color, 22)
            text(d, start, 839, f"tick {j}", 22, MUTED)
        arrow(d, [(455, 782), (525, 782)], V)
        text(
            d, 1010, 609, "Native D2 prompt wording is kept, including “audio question”.", 18, MUTED
        )

    def base_training(self, image, d):
        labels = ["User audio U[i]", "Delayed self A[i]", "Vision V[i]", "Previous text z*[i−1]"]
        for row, (name, color) in enumerate(zip(labels, COLORS)):
            text(d, 90, 306 + row * 78, name, 28, color, True)
        for column, tick in enumerate(range(10, 15)):
            x = 460 + column * 278
            center(d, x, 250, 230, f"t = {tick * 0.08:.2f} s", 24, MUTED)
            values = [
                f"U[{tick}]",
                f"E(S*[{tick - 1}])",
                "VIS PAD" if tick == 12 else f"v0[{tick}]" if tick < 12 else f"v1[{tick - 13}]",
                f"z*[{tick - 1}]",
            ]
            for row, (name, color) in enumerate(zip(values, COLORS)):
                pill(
                    d, (x, 290 + row * 78, 230, 59), name, MUTED if name == "VIS PAD" else color, 24
                )
        text(d, 90, 624, "SERIALIZE EACH TICK →", 24, WHITE, True)
        pill(d, (90, 677, 260, 65), "Prompt prefix", V, 25)
        for j, tick in enumerate(range(10, 13)):
            for k, (name, color) in enumerate(
                zip(
                    [f"U{tick}", f"A{tick}", "PAD" if tick == 12 else f"V{tick}", f"z*{tick - 1}"],
                    COLORS,
                )
            ):
                pill(d, (415 + j * 465 + k * 108, 677, 98, 65), name, color, 23)
            text(d, 415 + j * 465, 749, f"tick {tick}", 20, MUTED)
        card(
            d,
            (90, 798, 1735, 81),
            "Targets: current text z*[i] and current codec frame c*[i]",
            color=GREEN,
            size=29,
        )
        text(d, 1000, 624, "* = reference history / causal attention", 24, MUTED)

    def base_memory(self, image, d):
        text(
            d,
            90,
            255,
            "Concatenate same-task episodes to ≥720 s; the prompt starts the conversation once.",
            27,
            WHITE,
        )
        widths = [250, 345, 190, 390, 545]
        x = 90
        for j, width in enumerate(widths):
            pill(d, (x, 310, width - 15, 55), f"episode {j + 1}", V, 21)
            x += width
        for j in range(8):
            card(d, (90 + j * 218, 410, 200, 92), f"Chunk {j + 1}", "90 s", BORDER, 25, 21)
        card(
            d,
            (90, 565, 735, 200),
            "Previous 90 s / detached KV",
            "Forward context is available.\nNo gradient through this memory.",
            BORDER,
            32,
            29,
        )
        card(
            d,
            (965, 565, 860, 200),
            "Current 90 s / trainable",
            "1,125 ticks × 4 = 4,500 stream positions\nPrompt tokens are extra at sample start.",
            GREEN,
            32,
            28,
        )
        arrow(d, [(825, 665), (965, 665)], V)
        text(d, 829, 607, "carry", 21, V)
        pill(d, (1115, 805, 550, 60), "Backpropagate current chunk only", GREEN, 25)
        arrow(d, [(1390, 805), (1390, 765)], GREEN)
        text(d, 95, 812, "Carry encoder, text, codec,\nTalker and playback state.", 26, MUTED)

    def base_objectives(self, image, d):
        card(
            d,
            (90, 275, 595, 195),
            "Text cross entropy",
            "Predict current text / control tokens.\nRESPONSE and INTERRUPT weight: ×20",
            T,
            32,
            26,
        )
        card(
            d,
            (760, 275, 595, 195),
            "Speech-code cross entropy",
            "Predict the current codec frame.\nSpeech-code EOS weight: ×5",
            A,
            30,
            26,
        )
        card(
            d,
            (1450, 300, 375, 145),
            "Native D2 SFT groups",
            "Vision encoder stays frozen",
            GREEN,
            29,
            24,
        )
        arrow(d, [(387, 470), (387, 520), (1415, 520), (1415, 370), (1450, 370)], T)
        arrow(d, [(1355, 370), (1450, 370)], A)
        text(d, 785, 486, "L = Ltext + Lcodec", 26, WHITE, True)
        card(
            d,
            (90, 565, 1735, 100),
            "RESPONSE / INTERRUPT targets",
            "RMS-derived anchor +6 frames (480 ms)",
            U,
            28,
            26,
        )
        card(
            d,
            (90, 705, 1735, 100),
            "Speech-code EOS target",
            "Last audible assistant frame +1 frame",
            A,
            28,
            26,
        )
        center(
            d,
            90,
            849,
            1735,
            "Inference: 80-ms ticks    /    Training: 90-s chunks    /    Same [U, A, V, Tprev] order",
            28,
            WHITE,
            True,
        )

    def frame(self, index, elapsed, duration):
        name = self.book["scenes"][index]["id"]
        image = self.backgrounds[name].copy()
        d = ImageDraw.Draw(image)
        p = elapsed / duration
        if name == "overview":
            for j, color in enumerate(COLORS):
                if j < 3:
                    packet(
                        d,
                        [(1390, 335 + j * 118), (1450, 335 + j * 118), (1450, 460), (1495, 460)],
                        elapsed * 0.3 + j * 0.23,
                        color,
                    )
            highlight(d, (995, 290 + int(p * 8) % 4 * 118, 395, 90), COLORS[int(p * 8) % 4])
        elif name == "audio":
            short_wave(d, (115, 407, 260, 50), elapsed, U)
            short_wave(d, (115, 707, 260, 50), elapsed - 0.5, A)
            for y, color in [(400, U), (700, A)]:
                packet(d, [(745, y), (820, y)], elapsed * 0.5, color)
                packet(d, [(1385, y), (1470, y)], elapsed * 0.5 - 0.4, color)
            highlight(
                d,
                (1470, 360 if int(elapsed * 1.5) % 2 == 0 else 650, 335, 115),
                U if int(elapsed * 1.5) % 2 == 0 else A,
            )
        elif name == "vision":
            tick = min(24, int((p % 1) * 25))
            box = (88 + tick * 70, 705, 60, 67)
            highlight(d, box, WHITE)
            text(
                d,
                110,
                600,
                f"Illustrated time: {tick * 80:4} ms  /  next visual position: "
                + (
                    "PAD"
                    if tick == 12
                    else f"frame {0 if tick < 12 else 1}, token {tick if tick < 12 else tick - 13}"
                ),
                25,
                WHITE,
            )
            if tick != 12:
                highlight(
                    d,
                    (1345 + (tick % 13) % 4 * 115, 345 + ((tick % 13) // 4) * 62, 98, 47),
                    WHITE,
                    3,
                )
        elif name == "inference":
            clock = min(p * 4, 3.999999)
            tick = int(clock)
            phase = int((clock - tick) * 4)
            previous = "PAD" if tick == 0 else f"z[{tick - 1}]"
            labels = [
                f"U[{tick}]",
                "A[0] / silence" if tick == 0 else f"A[{tick}]",
                f"V[{tick}]",
                previous,
            ]
            for j, (label, color) in enumerate(zip(labels, COLORS)):
                pill(
                    d,
                    (145 + j * 425, 445, 350, 100),
                    label,
                    color,
                    35 if j == 1 and tick == 0 else 41,
                )
            text(
                d,
                130,
                340,
                f"Tick {tick} / media time {tick * 80} ms / each tick is slowed for explanation",
                23,
                MUTED,
            )
            card(
                d,
                (100, 675, 290, 110),
                f"KV before tick {tick}",
                f"P + {tick * 4} positions",
                BORDER,
                27,
                23,
            )
            appended = 4 * tick + (0 if phase == 0 else 3 if phase == 1 else 4)
            card(
                d,
                (530, 640, 850, 180),
                "Thinker / shared causal KV cache",
                f"Append [U[{tick}], A[{tick}], V[{tick}], {previous}]\nCurrent cache: P prompt tokens + {appended} stream positions",
                U,
                34,
                26,
            )
            card(d, (1510, 655, 310, 140), f"Predict z[{tick}]", "Text or control token", T, 33, 23)
            if phase in (0, 1):
                highlight(d, (125, 420, 1240, 155), U)
                packet(d, [(720, 585), (720, 640)], elapsed * 0.6, U)
                if phase == 1:
                    fraction = (clock - tick) * 4 - 1
                    for j, color in enumerate(COLORS[:3]):
                        x = (320 + j * 425) * (1 - fraction) + (860 + j * 90) * fraction
                        y = 557 * (1 - fraction) + 622 * fraction
                        pill(d, (x - 42, y - 17, 84, 34), f"{['U', 'A', 'V'][j]}{tick}", color, 19)
            else:
                highlight(d, (1420, 445, 350, 100), T)
                packet(d, [(1595, 545), (1595, 595), (1200, 595), (1200, 640)], elapsed * 0.45, T)
                packet(d, [(1380, 730), (1510, 730)], elapsed * 0.45, T)
            if phase == 3:
                highlight(d, (1510, 655, 310, 140), T)
        elif name == "feedback":
            packet(d, [(1330, 375), (1430, 375)], elapsed * 0.4, A)
            packet(
                d,
                [(1625, 450), (1625, 610), (1045, 610), (575, 610), (265, 610), (265, 465)],
                elapsed * 0.12,
                A,
            )
            x = 355 + 1460 * p
            d.line((x, 700, x, 860), fill=WHITE, width=3)
            if p > 0.46:
                text(d, 1130, 763, "Talker still speaking", 25, A, True)
        elif name == "prompts":
            if p < 0.7:
                highlight(d, (90 if p < 0.35 else 990, 265, 840, 370), U if p < 0.35 else V)
            else:
                packet(d, [(455, 782), (525, 782)], elapsed * 0.5, V)
                j = min(2, int((p - 0.7) / 0.3 * 3))
                highlight(d, (535 + j * 425, 737, 353, 82), U)
        elif name == "training":
            column = min(2, int(p * 3))
            highlight(d, (450 + column * 278, 280, 250, 315), COLORS[column % 4])
            if column < 3:
                highlight(d, (405 + column * 465, 667, 432, 85), U)
                packet(d, [(575 + column * 100, 605), (575 + column * 100, 655)], elapsed * 0.7, U)
        elif name == "memory":
            chunk = min(7, 1 + int(p * 7))
            highlight(d, (90 + (chunk - 1) * 218, 410, 200, 92), V, 3)
            highlight(d, (90 + chunk * 218, 410, 200, 92), GREEN, 4)
            packet(d, [(825, 665), (965, 665)], elapsed * 0.4, V)
            packet(d, [(1390, 805), (1390, 765)], elapsed * 0.6, GREEN)
            d.line((880, 710, 910, 740), fill=RED, width=4)
            d.line((910, 710, 880, 740), fill=RED, width=4)
        elif name == "objectives":
            highlight(
                d,
                [(90, 275, 595, 195), (760, 275, 595, 195), (1450, 300, 375, 145)][
                    min(2, int(p * 3))
                ],
                [T, A, GREEN][min(2, int(p * 3))],
            )
        return image


def prepare_preview(args, book, timeline):
    renderer = Diagrams(book)
    previews = args.cache / "previews"
    previews.mkdir(exist_ok=True)
    for index, scene in enumerate(timeline["scenes"]):
        duration = scene["end"] - scene["start"]
        frame = renderer.frame(index, duration * 0.70, duration)
        frame.save(previews / f"{index + 1:02}-{scene['id']}.png")
    cover = renderer.frame(0, 3, 10)
    cover.save(args.output / "cover.png")
    return renderer


def render(args, book, timeline):
    renderer = prepare_preview(args, book, timeline)
    metadata = [";FFMETADATA1", f"title={book['title']}", "artist=D2-Omni team"]
    for scene in timeline["scenes"]:
        metadata += [
            "[CHAPTER]",
            "TIMEBASE=1/1000",
            f"START={round(scene['start'] * 1000)}",
            f"END={round(scene['end'] * 1000)}",
            f"title={scene['title']}",
        ]
    meta_path = args.cache / "chapters.ffmetadata"
    meta_path.write_text("\n".join(metadata) + "\n")
    output = args.output / "d2-omni-architecture.mp4"
    command = [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgb24",
        "-video_size",
        f"{W}x{H}",
        "-framerate",
        str(FPS),
        "-i",
        "pipe:0",
        "-i",
        str(args.cache / "narration.wav"),
        "-f",
        "ffmetadata",
        "-i",
        str(meta_path),
        "-map",
        "0:v",
        "-map",
        "1:a",
        "-map_metadata",
        "2",
        "-map_chapters",
        "2",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-threads",
        "4",
        "-c:a",
        "aac",
        "-b:a",
        "160k",
        "-ar",
        "48000",
        "-af",
        "loudnorm=I=-16:TP=-1.5:LRA=11",
        "-t",
        str(timeline["duration"]),
        "-movflags",
        "+faststart",
        str(output),
    ]
    (args.cache / "ffmpeg-command.json").write_text(json.dumps(command, indent=2) + "\n")
    frames = round(timeline["duration"] * FPS)
    index, cue_index = 0, 0
    cues, scenes = timeline["cues"], timeline["scenes"]
    with (args.cache / "render.log").open("w") as log:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=log, stderr=log)
        try:
            for number in range(frames):
                now = number / FPS
                while index + 1 < len(scenes) and now >= scenes[index]["end"]:
                    index += 1
                scene = scenes[index]
                elapsed, duration = now - scene["start"], scene["end"] - scene["start"]
                image = renderer.frame(index, elapsed, duration)
                # Short visual transitions happen inside the narration pauses.
                fade = min(1, elapsed / 0.25, (duration - elapsed) / 0.25)
                if fade < 1:
                    image = Image.blend(Image.new("RGB", (W, H), BG), image, max(0, fade))
                d = ImageDraw.Draw(image)
                while cue_index + 1 < len(cues) and now >= cues[cue_index]["end"]:
                    cue_index += 1
                cue = cues[cue_index]
                if cue["start"] <= now < cue["end"]:
                    d.rounded_rectangle((65, 978, 1855, 1061), radius=15, fill="#07101c")
                    lines = wrap(cue["text"], 28, 1700)
                    if len(lines) > 2:
                        raise ValueError("Caption exceeds two lines")
                    y = 990 + (2 - len(lines)) * 17
                    for line in lines:
                        center(d, 85, y, 1750, line, 28, WHITE)
                        y += 34
                d.rectangle((0, 1074, W * now / timeline["duration"], 1080), fill=U)
                process.stdin.write(image.tobytes())
                if number % (FPS * 10) == 0:
                    print(
                        json.dumps(
                            dict(
                                event="render",
                                second=round(now),
                                scene=scene["id"],
                                total_seconds=round(timeline["duration"]),
                            )
                        ),
                        flush=True,
                    )
            process.stdin.close()
            if process.wait() != 0:
                raise RuntimeError((args.cache / "render.log").read_text()[-4000:])
        except BaseException:
            process.kill()
            process.wait()
            raise
    print(
        json.dumps(dict(event="video_complete", path=str(output), bytes=output.stat().st_size)),
        flush=True,
    )


def main(args):
    book = json.loads(args.storyboard.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    args.cache.mkdir(parents=True, exist_ok=True)
    (args.output / "sources.json").write_text(json.dumps(validate_sources(book), indent=2) + "\n")
    if args.stage in {"all", "narrate"}:
        narrate(args, book)
    if args.stage in {"all", "render", "preview"}:
        timeline = json.loads((args.cache / "timeline.json").read_text())
        if args.stage == "preview":
            prepare_preview(args, book, timeline)
        else:
            render(args, book, timeline)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["all", "narrate", "render", "preview"], default="all")
    parser.add_argument(
        "--storyboard", type=Path, default=Path("demo/architecture/storyboard.json")
    )
    parser.add_argument("--output", type=Path, default=Path("demo/architecture"))
    parser.add_argument("--cache", type=Path, default=Path(".cache/architecture-video"))
    parser.add_argument(
        "--voice", type=Path, default=Path(".cache/architecture-voice/en_US-lessac-high.onnx")
    )
    main(parser.parse_args())
