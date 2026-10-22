import unittest

import torch
from d2_qwen.model.duplex import (
    pack_macro_triplets,
    select_macro_text_slots,
    macro_attention_mask,
    macro_position_ids,
    D2Qwen3OmniModel,
)
from d2_qwen.model.config import D2Config
from model.vision_stream import visual_schedule, align_visual_features


class VisualStreamTests(unittest.TestCase):
    def test_causal_schedule_and_resume(self):
        image, token = visual_schedule(125)
        self.assertEqual(int((token == -1).sum()), 5)
        self.assertEqual(token[:25].tolist(), list(range(12)) + [-1] + list(range(12)))
        self.assertTrue(bool((image * 1000 <= torch.arange(125) * 80).all()))
        for start in (1, 12, 13, 24, 25, 60):
            a, b = visual_schedule(125 - start, start_tick=start)
            torch.testing.assert_close(a, image[start:])
            torch.testing.assert_close(b, token[start:])

    def test_four_slot_pack_readout_and_causality(self):
        streams = [torch.full((1, 3, 2), float(i)) for i in range(4)]
        packed = pack_macro_triplets(
            streams[0], streams[1], streams[3], 1, vision_inputs=streams[2]
        )
        self.assertEqual(packed[0, :, 0].tolist(), [0, 1, 2, 3] * 3)
        torch.testing.assert_close(select_macro_text_slots(packed, 1, 4), streams[3])
        mask = macro_attention_mask(
            frame_mask=torch.ones(1, 3, dtype=torch.bool), frames_per_unit=1, stream_count=4
        )
        torch.testing.assert_close(mask[0, 0], torch.ones(12, 12, dtype=torch.bool).tril())
        positions = macro_position_ids(
            batch_size=1,
            frame_count=3,
            frames_per_unit=1,
            frame_offset=0,
            device=torch.device("cpu"),
            stream_count=4,
        )
        self.assertEqual(positions[0, 0].tolist(), list(range(12)))

    def test_alignment_keeps_pad_and_stops_at_video_end(self):
        features = torch.arange(24).reshape(2, 12, 1).float()
        batch = align_visual_features(features, [features + 100], 30)
        self.assertEqual(int(batch["vision_mask"].sum()), 24)
        self.assertFalse(bool(batch["vision_mask"][0, 12]))
        self.assertFalse(bool(batch["vision_mask"][0, 25:].any()))
        self.assertEqual(float(batch["vision_features"][0, 13, 0]), 12.0)

    def test_inference_uses_same_three_token_prefix_and_fourth_text_slot(self):
        model = D2Qwen3OmniModel.__new__(D2Qwen3OmniModel)
        torch.nn.Module.__init__(model)
        model.d2_config = D2Config(latency_ms=80)
        model._compute_dtype = torch.float32
        model.stream_type_embedding = torch.nn.Embedding(2, 4)
        model.self_silence_embed = torch.nn.Parameter(torch.zeros(4))
        model.enable_vision()
        model.embed_text = lambda ids: torch.zeros((*ids.shape, 4))
        audio = torch.zeros(1, 1, 4)
        visual = torch.ones(1, 1, 4)
        deepstack = torch.full((3, 1, 1, 4), 2.0)
        for tick in range(3):
            prepared = model.inference_audio_prefill(
                environment_audio=audio,
                self_audio=audio,
                previous_self_audio=None,
                unit_offset=tick,
                cached_tokens=tick * 4,
                vision_features=visual,
                vision_mask=torch.ones(1, 1, dtype=torch.bool),
                vision_deepstack=deepstack,
            )
            self.assertEqual(prepared.inputs_embeds.shape, (1, 3, 4))
            self.assertEqual(prepared.position_ids[0, 0].tolist(), [tick * 4 + j for j in range(3)])
            self.assertEqual(prepared.visual_pos_masks.tolist(), [[False, False, True]])
            self.assertEqual(len(prepared.deepstack_visual_embeds), 3)
            text = model.inference_text_step(
                previous_text_ids=torch.zeros(1, dtype=torch.long),
                text_index=0,
                unit_offset=tick,
                cached_tokens=tick * 4 + 3,
            )
            self.assertEqual(int(text.position_ids[0, 0, 0]), tick * 4 + 3)


if __name__ == "__main__":
    unittest.main()
