"""The fast path must match D2's complete mask and preserve padded fallback."""

from types import SimpleNamespace

import torch
import torch.nn.functional as F

from d2_qwen.model.duplex import (
    macro_attention_mask,
    macro_position_ids,
    prepend_system_prompt_to_macro_layout,
)
from scripts.causal_attention import attention_forward


def test_full_macro_and_prompt_masks_are_lower_right_causal():
    frames = torch.ones(1, 5, dtype=torch.bool)
    mask = macro_attention_mask(
        frame_mask=frames, frames_per_unit=1, stream_count=4, cached_tokens=20
    )
    expected = torch.arange(40)[None] <= torch.arange(20)[:, None] + 20
    torch.testing.assert_close(mask[0, 0], expected)
    first = macro_attention_mask(frame_mask=frames, frames_per_unit=1, stream_count=4)
    _, _, prefixed = prepend_system_prompt_to_macro_layout(
        prompt_embeddings=torch.zeros(1, 3, 8),
        macro_embeddings=torch.zeros(1, 20, 8),
        macro_positions=macro_position_ids(
            batch_size=1,
            frame_count=5,
            frames_per_unit=1,
            frame_offset=0,
            device=torch.device("cpu"),
            stream_count=4,
        ),
        macro_attention=first,
    )
    torch.testing.assert_close(prefixed[0, 0], torch.ones(23, 23, dtype=torch.bool).tril())


def test_causal_dispatch_and_padded_fallback_match_explicit_sdpa():
    torch.manual_seed(83)
    q = torch.randn(1, 4, 5, 16)
    k, v = torch.randn(1, 2, 10, 16), torch.randn(1, 2, 10, 16)
    mask = (torch.arange(10)[None] <= torch.arange(5)[:, None] + 5)[None, None]
    module = SimpleNamespace(_d2_full_causal=True, num_key_value_groups=2, is_causal=True)
    expected = F.scaled_dot_product_attention(
        q, k.repeat_interleave(2, dim=1), v.repeat_interleave(2, dim=1), attn_mask=mask
    )
    actual, _ = attention_forward(module, q, k, v, mask)
    torch.testing.assert_close(actual, expected.transpose(1, 2))
    module._d2_full_causal = False
    mask[..., 3] = False
    expected = F.scaled_dot_product_attention(
        q, k.repeat_interleave(2, dim=1), v.repeat_interleave(2, dim=1), attn_mask=mask
    )
    actual, _ = attention_forward(module, q, k, v, mask)
    torch.testing.assert_close(actual, expected.transpose(1, 2))
