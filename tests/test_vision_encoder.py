"""Small CPU checks: upstream parity, gradients, and checkpoint extraction."""

import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeVisionEncoder as UpstreamVisionEncoder,
)

from model.vision_encoder import Qwen3OmniMoeVisionEncoder, Qwen3OmniMoeVisionEncoderConfig
from scripts.export_vision_encoder import export_vision_encoder


def tiny_config(attention="eager"):
    config = Qwen3OmniMoeVisionEncoderConfig(
        depth=2,
        hidden_size=32,
        intermediate_size=64,
        num_heads=4,
        in_channels=3,
        patch_size=2,
        temporal_patch_size=2,
        spatial_merge_size=2,
        out_hidden_size=48,
        num_position_embeddings=16,
        deepstack_visual_indexes=[0, 1],
    )
    config._attn_implementation = attention
    return config


class VisionEncoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_matches_upstream_outputs_and_gradients(self):
        for attention in ("eager", "sdpa"):
            with self.subTest(attention=attention):
                torch.manual_seed(0)
                upstream = UpstreamVisionEncoder(tiny_config(attention)).eval()
                local = Qwen3OmniMoeVisionEncoder(tiny_config(attention)).eval()
                local.load_state_dict(upstream.state_dict(), strict=True)
                # Two temporal patches plus a separate image: also exercises packing.
                grid = torch.tensor([[2, 4, 4], [1, 4, 4]])
                pixels = torch.randn(48, 3 * 2 * 2 * 2)
                expected = upstream(pixels, grid, return_dict=True)
                actual = local(pixels, grid, return_dict=True)
                torch.testing.assert_close(actual.last_hidden_state, expected.last_hidden_state)
                torch.testing.assert_close(actual.pooler_output, expected.pooler_output)
                self.assertEqual(actual.pooler_output.shape, (12, 48))
                self.assertEqual(len(actual.deepstack_features), 2)
                for a, b in zip(actual.deepstack_features, expected.deepstack_features):
                    torch.testing.assert_close(a, b)
                local_loss = actual.pooler_output.square().sum()
                upstream_loss = expected.pooler_output.square().sum()
                for a, b in zip(actual.deepstack_features, expected.deepstack_features):
                    local_loss = local_loss + a.square().sum()
                    upstream_loss = upstream_loss + b.square().sum()
                local_loss.backward()
                upstream_loss.backward()
                for name, param in local.named_parameters():
                    self.assertIsNotNone(param.grad, name)
                    torch.testing.assert_close(param.grad, dict(upstream.named_parameters())[name].grad)

    def test_export_and_reload_sharded_and_single_file_checkpoints(self):
        for sharded in (False, True):
            with self.subTest(sharded=sharded), tempfile.TemporaryDirectory() as tmp:
                checkpoint = Path(tmp) / "qwen"
                checkpoint.mkdir()
                encoder = Qwen3OmniMoeVisionEncoder(tiny_config()).eval()
                config = {"thinker_config": {"vision_config": encoder.config.to_dict()}}
                (checkpoint / "config.json").write_text(json.dumps(config))
                tensors = {f"thinker.visual.{k}": v for k, v in encoder.state_dict().items()}
                tensors["thinker.unrelated.weight"] = torch.ones(1)
                if sharded:
                    items = list(tensors.items())
                    weight_map = {}
                    for i, subset in enumerate((items[::2], items[1::2])):
                        filename = f"model-{i}.safetensors"
                        save_file(dict(subset), checkpoint / filename)
                        weight_map.update({key: filename for key, _ in subset})
                    (checkpoint / "model.safetensors.index.json").write_text(
                        json.dumps({"weight_map": weight_map})
                    )
                else:
                    save_file(tensors, checkpoint / "model.safetensors")
                output = Path(tmp) / "vision"
                count = export_vision_encoder(checkpoint, output)
                self.assertEqual(count, len(encoder.state_dict()))
                restored, info = Qwen3OmniMoeVisionEncoder.from_pretrained(
                    output, attn_implementation="eager", output_loading_info=True
                )
                self.assertFalse(info["missing_keys"])
                self.assertFalse(info["unexpected_keys"])
                pixels = torch.randn(16, 24)
                grid = torch.tensor([[1, 4, 4]])
                torch.testing.assert_close(
                    restored(pixels, grid).pooler_output, encoder(pixels, grid).pooler_output
                )
                with self.assertRaises(FileExistsError):
                    export_vision_encoder(checkpoint, output)


if __name__ == "__main__":
    unittest.main()
