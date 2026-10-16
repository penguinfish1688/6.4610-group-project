"""Export only thinker.visual weights from a local Qwen3-Omni snapshot.

Usage: python scripts/export_vision_encoder.py --checkpoint /path/to/qwen --output checkpoints/vision_encoder
This copies tensors on CPU; it does not load the 30B model or download weights.
"""

import argparse
import json
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


def export_vision_encoder(checkpoint: Path, output: Path) -> int:
    config = json.loads((checkpoint / "config.json").read_text())
    vision_config = config["thinker_config"]["vision_config"]
    prefix = "thinker.visual."
    index_path = checkpoint / "model.safetensors.index.json"
    if index_path.is_file():
        weight_map = json.loads(index_path.read_text())["weight_map"]
        shards = sorted({v for k, v in weight_map.items() if k.startswith(prefix)})
    else:
        shards = ["model.safetensors"]
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output must be new or empty: {output}")
    tensors = {}
    for shard in shards:
        with safe_open(checkpoint / shard, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key.startswith(prefix):
                    name = key.removeprefix(prefix)
                    if name in tensors:
                        raise ValueError(f"Duplicate vision tensor: {key}")
                    tensors[name] = handle.get_tensor(key)
    if not tensors:
        raise ValueError(f"No {prefix} tensors in {checkpoint}")
    if index_path.is_file():
        expected = {key.removeprefix(prefix) for key in weight_map if key.startswith(prefix)}
        if tensors.keys() != expected:
            raise ValueError("Vision tensors do not match the checkpoint index")
    output.mkdir(parents=True, exist_ok=True)
    vision_config["architectures"] = ["Qwen3OmniMoeVisionEncoder"]
    (output / "config.json").write_text(json.dumps(vision_config, indent=2) + "\n")
    save_file(tensors, output / "model.safetensors", metadata={"format": "pt"})
    return len(tensors)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    count = export_vision_encoder(args.checkpoint, args.output)
    print(f"Exported {count} vision tensors to {args.output}")


if __name__ == "__main__":
    main()
