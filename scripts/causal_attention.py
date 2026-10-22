"""Use native FlashAttention for full causal D2 chunks, including cached K/V."""

import torch
import torch.nn.functional as F
from torch.nn.attention.bias import causal_lower_right
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS


def causal_attention(query, key, value, *, dropout=0.0, scaling=None):
    return F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=causal_lower_right(query.shape[-2], key.shape[-2]),
        dropout_p=dropout,
        scale=scaling,
        enable_gqa=query.shape[1] != key.shape[1],
    )


def attention_forward(
    module, query, key, value, attention_mask, dropout=0.0, scaling=None, **kwargs
):
    if getattr(module, "_d2_full_causal", False):
        output = causal_attention(query, key, value, dropout=dropout, scaling=scaling)
        return output.transpose(1, 2).contiguous(), None
    return sdpa_attention_forward(
        module, query, key, value, attention_mask, dropout=dropout, scaling=scaling, **kwargs
    )


def install(model):
    ALL_ATTENTION_FUNCTIONS.register("sdpa", attention_forward)
    set_causal_path(model, False)


def set_causal_path(model, enabled):
    if model.frames_per_unit != 1:
        raise ValueError("Full causal serialization is audited for the 80-ms model")
    model._d2_using_fast_causal = bool(enabled)
    for transformer in (model.qwen.thinker.model, model.qwen.talker.model):
        for layer in transformer.layers:
            layer.self_attn._d2_full_causal = bool(enabled)


def validate(device):
    """Check GQA forward/backward against the explicit lower-right boolean mask."""
    errors = []
    generator = torch.Generator(device=device).manual_seed(71)
    for query_length, key_length in ((64, 64), (64, 128), (65, 130)):
        inputs = [
            torch.randn(
                1, heads, length, 64, device=device, dtype=torch.bfloat16, generator=generator
            ).requires_grad_()
            for heads, length in ((4, query_length), (2, key_length), (2, key_length))
        ]
        copies = [value.detach().clone().requires_grad_() for value in inputs]
        mask = torch.arange(key_length, device=device)[None] <= (
            torch.arange(query_length, device=device)[:, None] + key_length - query_length
        )
        reference = F.scaled_dot_product_attention(
            inputs[0],
            inputs[1].repeat_interleave(2, dim=1),
            inputs[2].repeat_interleave(2, dim=1),
            attn_mask=mask,
        )
        actual = causal_attention(*copies)
        gradient = torch.randn(
            actual.shape, device=device, dtype=torch.bfloat16, generator=generator
        )
        reference.backward(gradient)
        actual.backward(gradient)
        torch.testing.assert_close(actual, reference, atol=0.005, rtol=0.03)
        for left, right in zip(inputs, copies, strict=True):
            torch.testing.assert_close(left.grad, right.grad, atol=0.02, rtol=0.04)
        errors.append(
            dict(
                query_length=query_length,
                key_length=key_length,
                maximum_output_error=float((actual - reference).detach().abs().max()),
            )
        )
    return errors
