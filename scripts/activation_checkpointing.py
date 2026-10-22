"""Recompute decoder activations with replay-safe Transformer-XL caches."""

from types import MethodType

import torch
from torch.utils.checkpoint import checkpoint


class ReplaySafeCache:
    """Write current K/V once; recomputation reuses the original frozen prefix."""

    def __init__(self, cache):
        self.cache = cache
        self.prefixes = {}

    def update(self, keys, values, layer_idx, *args, **kwargs):
        if layer_idx not in self.prefixes:
            layers = self.cache.layers
            layer = layers[layer_idx] if layer_idx < len(layers) else None
            prefix = (
                (layer.keys, layer.values)
                if layer is not None and getattr(layer, "keys", None) is not None
                else (None, None)
            )
            self.prefixes[layer_idx] = prefix
            return self.cache.update(keys, values, layer_idx, *args, **kwargs)
        old_keys, old_values = self.prefixes[layer_idx]
        if old_keys is None or old_keys.numel() == 0:
            return keys, values
        return (torch.cat((old_keys, keys), dim=-2), torch.cat((old_values, values), dim=-2))

    def __getattr__(self, name):
        return getattr(self.cache, name)


def checkpoint_attention(module):
    original = module.forward

    def forward(this, *args, **kwargs):
        if not torch.is_grad_enabled():
            return original(*args, **kwargs)
        cache = kwargs.get("past_key_values")
        if cache is not None:
            kwargs = dict(kwargs, past_key_values=ReplaySafeCache(cache))
        return checkpoint(original, *args, use_reentrant=False, preserve_rng_state=True, **kwargs)

    module.forward = MethodType(forward, module)


def checkpoint_mlp(module):
    original = module.forward

    def forward(this, *args, **kwargs):
        # D2 keeps the decoders in eval mode during SFT to disable dropout.
        # Gradient mode, rather than module.training, identifies training here.
        if torch.is_grad_enabled():
            return checkpoint(
                original, *args, use_reentrant=False, preserve_rng_state=True, **kwargs
            )
        return original(*args, **kwargs)

    module.forward = MethodType(forward, module)


def enable(model):
    count = 0
    for transformer in (model.qwen.thinker.model, model.qwen.talker.model):
        for layer in transformer.layers:
            checkpoint_mlp(layer.mlp)
            checkpoint_attention(layer.self_attn)
            count += 1
    return count
