"""MLP recomputation must preserve outputs, input gradients and adapter gradients."""

import copy

import torch
import pytest
from transformers.cache_utils import DynamicCache

from scripts.activation_checkpointing import checkpoint_mlp, checkpoint_attention


def test_mlp_checkpoint_gradient_parity():
    torch.manual_seed(29)
    reference = torch.nn.Sequential(torch.nn.Linear(8, 19), torch.nn.SiLU(), torch.nn.Linear(19, 8))
    candidate = copy.deepcopy(reference)
    checkpoint_mlp(candidate)
    candidate.eval()  # Native D2 disables decoder dropout during SFT.
    left = torch.randn(2, 7, 8, requires_grad=True)
    right = left.detach().clone().requires_grad_()
    expected, actual = reference(left), candidate(right)
    torch.testing.assert_close(actual, expected)
    expected.square().sum().backward()
    actual.square().sum().backward()
    torch.testing.assert_close(right.grad, left.grad)
    for a, b in zip(candidate.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(a.grad, b.grad)
    candidate.eval()
    with torch.no_grad():
        torch.testing.assert_close(candidate(right), reference(left))


@pytest.mark.parametrize("prefix_length", [0, 3])
def test_attention_replay_preserves_gradients_and_writes_cache_once(prefix_length):
    class Attention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.qkv = torch.nn.Linear(8, 24, bias=False)

        def forward(self, hidden_states, *, past_key_values):
            q, k, v = self.qkv(hidden_states).chunk(3, dim=-1)
            q, k, v = (item[:, None] for item in (q, k, v))
            k, v = past_key_values.update(k, v, 0)
            prefix = k.shape[-2] - q.shape[-2]
            mask = torch.arange(k.shape[-2])[None] <= torch.arange(q.shape[-2])[:, None] + prefix
            result = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
            return result[:, 0], None

    torch.manual_seed(41)
    reference = Attention()
    candidate = copy.deepcopy(reference)
    checkpoint_attention(candidate)
    caches = [DynamicCache(), DynamicCache()]
    if prefix_length:
        prefix = torch.randn(1, 1, prefix_length, 8)
        for cache in caches:
            cache.update(prefix.clone(), prefix.clone() * 2, 0)
    left = torch.randn(1, 4, 8, requires_grad=True)
    right = left.detach().clone().requires_grad_()
    expected, _ = reference(left, past_key_values=caches[0])
    actual, _ = candidate(right, past_key_values=caches[1])
    saved = caches[1].layers[0].keys.detach().clone()
    # Native XL compaction detaches the cache before the optimizer backward.
    for cache in caches:
        cache.layers[0].keys = cache.layers[0].keys.detach()
        cache.layers[0].values = cache.layers[0].values.detach()
    expected.square().sum().backward()
    actual.square().sum().backward()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(right.grad, left.grad)
    torch.testing.assert_close(candidate.qkv.weight.grad, reference.qkv.weight.grad)
    assert caches[1].get_seq_length() == prefix_length + 4
    torch.testing.assert_close(caches[1].layers[0].keys, saved)
