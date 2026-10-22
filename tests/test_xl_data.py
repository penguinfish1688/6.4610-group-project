import tempfile
from pathlib import Path
import unittest

import torch
from transformers.cache_utils import DynamicCache
from d2_qwen.model.base import QwenDuplexTransformerXLState
from model.vision_stream import align_visual_features
from scripts.xl_data import chunk_tensors, move_state


class XLDataTests(unittest.TestCase):
    def test_chunk_boundary_keeps_visual_phase_and_masks_tail(self):
        features = torch.arange(24).reshape(2, 12, 1).float()
        ticks = 30
        payload = dict(
            ticks=ticks,
            vision_bank=dict(features=features, deepstack=[features], repeat=True),
            tensors=dict(
                env_mel=torch.randn(1, 128, ticks * 8),
                self_mel=torch.randn(1, 128, ticks * 8),
                text_target=torch.arange(ticks)[None],
                codec_target=torch.zeros(1, ticks, 16, dtype=torch.long),
                frame_mask=torch.ones(1, ticks, dtype=torch.bool),
                assistant_mask=torch.zeros(1, ticks, dtype=torch.bool),
                system_prompt_input_ids=torch.tensor([[1, 2, 3]]),
            ),
        )
        first = chunk_tensors(payload, 0, 16, 999)
        second = chunk_tensors(payload, 16, 16, 999)
        self.assertIn("system_prompt_input_ids", first)
        self.assertNotIn("system_prompt_input_ids", second)
        self.assertEqual(second["frame_mask"].sum(), 14)
        self.assertEqual(second["text_target"][0, -2:].tolist(), [999, 999])
        expected = align_visual_features(features, [features], ticks, repeat=True)
        torch.testing.assert_close(
            second["vision_features"][:, :14], expected["vision_features"][:, 16:]
        )
        self.assertFalse(bool(second["vision_mask"][0, -2:].any()))

    def test_detached_recurrence_roundtrip_preserves_cache_and_audio_state(self):
        cache = DynamicCache()
        keys = torch.randn(1, 2, 4, 8, requires_grad=True)
        cache.update(keys, keys * 2, 0)
        state = QwenDuplexTransformerXLState(
            thinker_cache=cache, frame_offset=9, previous_self_audio=torch.ones(1, 1, 8)
        )
        saved = move_state(state, "cpu")
        self.assertFalse(saved.thinker_cache.layers[0].keys.requires_grad)
        self.assertIsNot(saved.thinker_cache, cache)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "state.pt"
            torch.save(saved, path)
            restored = move_state(torch.load(path, weights_only=False), "cpu")
        self.assertEqual(restored.frame_offset, 9)
        torch.testing.assert_close(restored.thinker_cache.layers[0].keys, keys.detach())
        torch.testing.assert_close(restored.previous_self_audio, state.previous_self_audio)


if __name__ == "__main__":
    unittest.main()
