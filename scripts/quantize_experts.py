"""Use D2's frozen INT8 expert format before sharding the remaining weights."""

import importlib
from pathlib import Path
import sys

import torch


@torch.no_grad()
def quantize(model, cfg, device, *, training=True):
    # Codec targets are prepared offline; training never renders waveforms.
    # Inference reloads this unchanged frozen vocoder from the base checkpoint.
    vocoder = model.qwen.code2wav
    if any(p.requires_grad for p in vocoder.parameters()):
        raise ValueError("The waveform renderer must remain frozen")
    unused_vocoder_bytes = 0
    if training:
        unused_vocoder_bytes = sum(p.numel() * p.element_size() for p in vocoder.parameters())
        del model.qwen.code2wav
    del vocoder
    reference = Path(cfg["legacy_d2_root"]) / "qwen_duplex"
    sys.path.insert(0, cfg["legacy_d2_root"])
    sys.path.insert(0, str(reference))
    legacy = importlib.import_module("qwen3.model.qwen3_omni.int8_moe")
    kernels = importlib.import_module("duplex.frozen_moe_int8")

    # The legacy SM8 inference fallback is intended for one streaming token.
    # Full 90-second training/validation batches use D2's grouped kernel.
    def grouped_linear(inputs, quantized, scales, offsets, *, compute_dtype):
        if quantized.dtype == torch.bfloat16:
            quantized = quantized.view(torch.int8)
        return legacy._REFERENCE_FROZEN_INT8_GROUPED_LINEAR(
            inputs, quantized, scales, offsets, compute_dtype=compute_dtype
        )

    if training:
        kernels._frozen_int8_grouped_linear = grouped_linear
    else:
        kernels._frozen_int8_grouped_linear = legacy._d2_frozen_int8_grouped_linear
    total_bytes, sharded_bytes, counts = 0, 0, []
    shard_tail = (
        cfg.get("shard_frozen_expert_layers", {"thinker": 8, "talker": 20})
        if training
        else {"thinker": 0, "talker": 0}
    )
    for kind, transformer, expected in (
        ("thinker", model.qwen.thinker.model, 48),
        ("talker", model.qwen.talker.model, 20),
    ):
        count = 0
        for index, layer in enumerate(transformer.layers):
            source = layer.mlp.experts
            if any(p.requires_grad for p in source.parameters()):
                raise ValueError("Routed experts must remain frozen")
            # At most one BF16 expert layer is resident during conversion.
            source.to(device)
            layer.mlp.experts = kernels.FrozenInt8Experts.from_frozen(source, expert_chunk_size=8)
            total_bytes += layer.mlp.experts.storage_bytes()
            if index >= expected - shard_tail[kind]:
                # FSDP shards parameters, not buffers. Frozen INT8 parameters
                # retain their exact bytes and never enter the optimizer.
                experts = layer.mlp.experts
                for name in ("gate_up_proj_qweight", "down_proj_qweight"):
                    tensor = experts._buffers.pop(name)
                    sharded_bytes += tensor.numel() * tensor.element_size()
                    from scripts.int8_sharding import floating_storage

                    experts.register_parameter(name, floating_storage(tensor))
            del source
            count += 1
        if count != expected:
            raise ValueError("The frozen-expert inventory differs from D2")
        counts.append(count)
    from scripts.activation_checkpointing import enable
    from scripts.precision import configure

    checkpointed = enable(model) if training else 0
    norm_count = configure(model)
    from scripts.causal_attention import install

    install(model)
    return dict(
        thinker_layers=counts[0],
        talker_layers=counts[1],
        checkpointed_mlp_layers=checkpointed,
        checkpointed_attention_layers=checkpointed,
        unused_training_vocoder_bytes_removed=unused_vocoder_bytes,
        norms_preserving_activation_dtype=norm_count,
        storage_bytes=total_bytes,
        sharded_storage_bytes=sharded_bytes,
        format="symmetric_per_output_channel_int8",
    )
