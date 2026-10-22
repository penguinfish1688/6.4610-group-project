"""Transcribe inference input/output WAVs for an independent speech spot check."""

import argparse
import json
from pathlib import Path
import wave

import numpy as np
from scipy.signal import resample_poly
import sherpa_onnx


def read_audio(path):
    with wave.open(str(path), "rb") as stream:
        if stream.getnchannels() != 1 or stream.getsampwidth() != 2:
            raise ValueError("Expected mono PCM16 WAV")
        rate = stream.getframerate()
        samples = np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2")
    samples = samples.astype(np.float32) / 32768
    if rate == 24000:
        samples = resample_poly(samples, 2, 3).astype(np.float32)
    elif rate != 16000:
        raise ValueError(f"Unsupported WAV sample rate: {rate}")
    return samples


def run(args):
    root = args.model
    recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
        encoder=str(root / "encoder-epoch-99-avg-1-chunk-16-left-128.onnx"),
        decoder=str(root / "decoder-epoch-99-avg-1-chunk-16-left-128.onnx"),
        joiner=str(root / "joiner-epoch-99-avg-1-chunk-16-left-128.onnx"),
        tokens=str(root / "tokens.txt"),
        provider="cpu",
        num_threads=2,
        sample_rate=16000,
        feature_dim=80,
        model_type="zipformer2",
        decoding_method="greedy_search",
        enable_endpoint_detection=False,
    )
    records = []
    for result_path in sorted(args.run.glob("*/result.json")):
        result = json.loads(result_path.read_text())
        record = dict(name=result["name"], generated_text=result["generated_text"])
        for lane in ("input", "output"):
            samples = read_audio(result_path.parent / f"{lane}.wav")
            stream = recognizer.create_stream()
            stream.accept_waveform(16000, np.pad(samples, (0, 16000)))
            stream.input_finished()
            while recognizer.is_ready(stream):
                recognizer.decode_stream(stream)
            record[f"{lane}_asr"] = recognizer.get_result(stream).strip()
        records.append(record)
        print(json.dumps(record), flush=True)
    if not records:
        raise FileNotFoundError("No completed inference examples")
    output = dict(
        model=str(root),
        purpose="Automatic qualitative spot check, not a benchmark score",
        examples=records,
    )
    (args.run / "asr.json").write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("runs/inference-step1000"))
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(
            "/orcd/pool/006/chang168/LSKV/models/asr/sherpa-onnx-streaming-zipformer-en-2023-06-26-fp32"
        ),
    )
    run(parser.parse_args())
