# Vendored unchanged from the D2 experiment runtime for portable inference.
# See model/INT8_UPSTREAM.json for source paths and SHA256 provenance.
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


QUANTIZATION_SCHEME = "symmetric_per_output_channel"
IMPLEMENTATION = "frozen_int8_dequant_grouped_mm_autograd_v1"
QWEN3_OMNI_THINKER_EXPERT_MODULES = 48
QWEN3_OMNI_THINKER_EXPERT_PARAMETERS = 28_991_029_248


@dataclass(frozen=True)
class FrozenThinkerExpertsInt8Options:
    enabled: bool = False
    expected_modules: int = QWEN3_OMNI_THINKER_EXPERT_MODULES
    expected_parameters: int = QWEN3_OMNI_THINKER_EXPERT_PARAMETERS
    expert_chunk_size: int = 8
    quantization_scheme: str = QUANTIZATION_SCHEME

    @classmethod
    def from_model_config(
        cls,
        model_config: Mapping[str, Any],
    ) -> "FrozenThinkerExpertsInt8Options":
        raw = model_config.get("thinker_expert_weight_only_int8", False)
        if isinstance(raw, bool):
            return cls(enabled=raw)
        if not isinstance(raw, Mapping):
            raise TypeError(
                "model.thinker_expert_weight_only_int8 must be a bool or mapping"
            )
        allowed = {
            "enabled",
            "expected_modules",
            "expected_parameters",
            "expert_chunk_size",
            "quantization_scheme",
        }
        unexpected = sorted(set(raw) - allowed)
        if unexpected:
            raise ValueError(
                "Unknown model.thinker_expert_weight_only_int8 options: "
                f"{unexpected}"
            )
        enabled = raw.get("enabled", True)
        if not isinstance(enabled, bool):
            raise TypeError(
                "model.thinker_expert_weight_only_int8.enabled must be a bool"
            )
        options = cls(
            enabled=enabled,
            expected_modules=int(
                raw.get(
                    "expected_modules",
                    QWEN3_OMNI_THINKER_EXPERT_MODULES,
                )
            ),
            expected_parameters=int(
                raw.get(
                    "expected_parameters",
                    QWEN3_OMNI_THINKER_EXPERT_PARAMETERS,
                )
            ),
            expert_chunk_size=int(raw.get("expert_chunk_size", 8)),
            quantization_scheme=str(
                raw.get("quantization_scheme", QUANTIZATION_SCHEME)
            ),
        )
        if options.expected_modules < 1:
            raise ValueError("expected_modules must be positive")
        if options.expected_parameters < 1:
            raise ValueError("expected_parameters must be positive")
        if options.expert_chunk_size < 1:
            raise ValueError("expert_chunk_size must be positive")
        if options.quantization_scheme != QUANTIZATION_SCHEME:
            raise ValueError(
                "Unsupported Thinker expert INT8 quantization scheme "
                f"{options.quantization_scheme!r}"
            )
        return options


def _quantize_per_output_channel(
    weight: torch.Tensor,
    *,
    expert_chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if weight.ndim != 3:
        raise ValueError(
            f"Packed expert weight must be 3-D, got {tuple(weight.shape)}"
        )
    quantized = torch.empty_like(weight, dtype=torch.int8)
    scales = torch.empty(
        (*weight.shape[:-1], 1),
        dtype=torch.float32,
        device=weight.device,
    )
    chunk_size = max(1, int(expert_chunk_size))
    for start in range(0, int(weight.shape[0]), chunk_size):
        end = min(int(weight.shape[0]), start + chunk_size)
        values = weight[start:end].float()
        chunk_scales = values.abs().amax(dim=-1, keepdim=True) / 127.0
        chunk_scales.masked_fill_(chunk_scales == 0.0, 1.0)
        quantized[start:end].copy_(
            torch.round(values / chunk_scales)
            .clamp_(-127.0, 127.0)
            .to(torch.int8)
        )
        scales[start:end].copy_(chunk_scales)
    return quantized, scales


def _dequantize_per_output_channel(
    quantized: torch.Tensor,
    scales: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    if quantized.dtype != torch.int8:
        raise TypeError(f"Expected INT8 expert weights, got {quantized.dtype}")
    weight = quantized.to(dtype=dtype)
    weight.mul_(scales.to(dtype=dtype))
    return weight


class _FrozenInt8GroupedLinear(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        inputs: torch.Tensor,
        quantized: torch.Tensor,
        scales: torch.Tensor,
        offsets: torch.Tensor,
        compute_dtype: torch.dtype,
    ) -> torch.Tensor:
        ctx.save_for_backward(quantized, scales, offsets)
        ctx.compute_dtype = compute_dtype
        ctx.input_dtype = inputs.dtype
        weight = _dequantize_per_output_channel(
            quantized,
            scales,
            dtype=compute_dtype,
        )
        return F.grouped_mm(
            inputs.to(dtype=compute_dtype),
            weight.transpose(-2, -1),
            offs=offsets,
        )

    @staticmethod
    def backward(
        ctx: Any,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, None, None, None, None]:
        quantized, scales, offsets = ctx.saved_tensors
        weight = _dequantize_per_output_channel(
            quantized,
            scales,
            dtype=ctx.compute_dtype,
        )
        grad_inputs = F.grouped_mm(
            grad_output.to(dtype=ctx.compute_dtype).contiguous(),
            weight,
            offs=offsets,
        )
        return (
            grad_inputs.to(dtype=ctx.input_dtype),
            None,
            None,
            None,
            None,
        )


def _frozen_int8_grouped_linear(
    inputs: torch.Tensor,
    quantized: torch.Tensor,
    scales: torch.Tensor,
    offsets: torch.Tensor,
    *,
    compute_dtype: torch.dtype,
) -> torch.Tensor:
    return _FrozenInt8GroupedLinear.apply(
        inputs,
        quantized,
        scales,
        offsets,
        compute_dtype,
    )


class FrozenInt8Experts(nn.Module):
    def __init__(
        self,
        *,
        gate_up_proj_qweight: torch.Tensor,
        gate_up_proj_scale: torch.Tensor,
        down_proj_qweight: torch.Tensor,
        down_proj_scale: torch.Tensor,
        act_fn: Any,
        config: Any,
        compute_dtype: torch.dtype,
    ) -> None:
        super().__init__()
        if compute_dtype not in (torch.bfloat16, torch.float16):
            raise TypeError(
                "Frozen INT8 experts require BF16 or FP16 compute, got "
                f"{compute_dtype}"
            )
        experts, two_intermediate, hidden = map(
            int,
            gate_up_proj_qweight.shape,
        )
        down_experts, down_hidden, intermediate = map(
            int,
            down_proj_qweight.shape,
        )
        if (
            experts != down_experts
            or hidden != down_hidden
            or two_intermediate != 2 * intermediate
        ):
            raise ValueError(
                "Inconsistent quantized expert shapes: "
                f"gate_up={tuple(gate_up_proj_qweight.shape)} "
                f"down={tuple(down_proj_qweight.shape)}"
            )
        expected_gate_scale = (*gate_up_proj_qweight.shape[:-1], 1)
        expected_down_scale = (*down_proj_qweight.shape[:-1], 1)
        if tuple(gate_up_proj_scale.shape) != expected_gate_scale:
            raise ValueError(
                "Invalid gate/up scale shape: "
                f"actual={tuple(gate_up_proj_scale.shape)} "
                f"expected={expected_gate_scale}"
            )
        if tuple(down_proj_scale.shape) != expected_down_scale:
            raise ValueError(
                "Invalid down scale shape: "
                f"actual={tuple(down_proj_scale.shape)} "
                f"expected={expected_down_scale}"
            )
        self.num_experts = experts
        self.hidden_dim = hidden
        self.intermediate_dim = intermediate
        self.act_fn = act_fn
        self.config = config
        self.compute_dtype = compute_dtype
        self.has_gate = True
        self.has_bias = False
        self.is_transposed = False
        self.is_concatenated = True
        self.register_buffer(
            "gate_up_proj_qweight",
            gate_up_proj_qweight,
            persistent=True,
        )
        self.register_buffer(
            "gate_up_proj_scale",
            gate_up_proj_scale,
            persistent=True,
        )
        self.register_buffer(
            "down_proj_qweight",
            down_proj_qweight,
            persistent=True,
        )
        self.register_buffer(
            "down_proj_scale",
            down_proj_scale,
            persistent=True,
        )

    @classmethod
    @torch.no_grad()
    def from_frozen(
        cls,
        experts: nn.Module,
        *,
        expert_chunk_size: int,
    ) -> "FrozenInt8Experts":
        parameters = tuple(experts.parameters())
        if not parameters:
            raise ValueError("Thinker expert module has no parameters")
        if any(parameter.requires_grad for parameter in parameters):
            raise RuntimeError(
                "Thinker routed experts must be frozen before INT8 conversion"
            )
        gate_up = getattr(experts, "gate_up_proj", None)
        down = getattr(experts, "down_proj", None)
        if not isinstance(gate_up, torch.Tensor) or not isinstance(
            down,
            torch.Tensor,
        ):
            raise TypeError(
                "Thinker expert module must expose gate_up_proj and down_proj "
                "tensors"
            )
        if gate_up.dtype != down.dtype:
            raise TypeError(
                "Thinker expert projection dtypes differ: "
                f"gate_up={gate_up.dtype} down={down.dtype}"
            )
        if gate_up.dtype not in (torch.bfloat16, torch.float16):
            raise TypeError(
                "Thinker expert source weights must be BF16 or FP16, got "
                f"{gate_up.dtype}"
            )
        gate_up_qweight, gate_up_scale = _quantize_per_output_channel(
            gate_up.detach(),
            expert_chunk_size=expert_chunk_size,
        )
        down_qweight, down_scale = _quantize_per_output_channel(
            down.detach(),
            expert_chunk_size=expert_chunk_size,
        )
        return cls(
            gate_up_proj_qweight=gate_up_qweight,
            gate_up_proj_scale=gate_up_scale,
            down_proj_qweight=down_qweight,
            down_proj_scale=down_scale,
            act_fn=getattr(experts, "act_fn"),
            config=getattr(experts, "config", None),
            compute_dtype=gate_up.dtype,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        device = hidden_states.device
        num_top_k = int(top_k_index.size(-1))
        num_tokens = int(hidden_states.size(0))
        sample_weights = top_k_weights.reshape(-1)
        expert_ids = top_k_index.reshape(-1)
        expert_ids_grouped, permutation = torch.sort(expert_ids)
        selected_hidden_states = hidden_states[permutation // num_top_k]
        sample_weights_grouped = sample_weights[permutation]
        histc_input = (
            expert_ids_grouped.float()
            if device.type in ("cpu", "mps")
            else expert_ids_grouped.int()
        )
        tokens_per_expert = torch.histc(
            histc_input,
            bins=self.num_experts,
            min=0,
            max=self.num_experts - 1,
        )
        offsets = torch.cumsum(
            tokens_per_expert,
            dim=0,
            dtype=torch.int32,
        )
        sentinel_mask = (
            expert_ids_grouped >= self.num_experts
        ).unsqueeze(-1)
        expert_ids_grouped.clamp_(max=self.num_experts - 1)
        selected_hidden_states.masked_fill_(sentinel_mask, 0.0)
        gate_up = _frozen_int8_grouped_linear(
            selected_hidden_states,
            self.gate_up_proj_qweight,
            self.gate_up_proj_scale,
            offsets,
            compute_dtype=self.compute_dtype,
        )
        gate, up = gate_up.chunk(2, dim=-1)
        intermediate = self.act_fn(gate) * up
        projected = _frozen_int8_grouped_linear(
            intermediate,
            self.down_proj_qweight,
            self.down_proj_scale,
            offsets,
            compute_dtype=self.compute_dtype,
        )
        weighted = projected * sample_weights_grouped.unsqueeze(-1)
        weighted.masked_fill_(sentinel_mask, 0.0)
        inverse_permutation = torch.empty_like(permutation)
        inverse_permutation[permutation] = torch.arange(
            permutation.size(0),
            device=device,
        )
        weighted = weighted[inverse_permutation]
        output = weighted.view(
            num_tokens,
            num_top_k,
            self.hidden_dim,
        ).sum(dim=1)
        return output.to(dtype=hidden_states.dtype)

    def storage_bytes(self) -> int:
        return sum(
            int(tensor.numel()) * int(tensor.element_size())
            for tensor in self.buffers()
        )


@torch.no_grad()
def quantize_frozen_thinker_experts_(
    model: nn.Module,
    options: FrozenThinkerExpertsInt8Options,
) -> dict[str, Any]:
    if not options.enabled:
        return {
            "enabled": False,
            "implementation": IMPLEMENTATION,
            "quantization_scheme": options.quantization_scheme,
        }
    thinker = getattr(getattr(model, "qwen", None), "thinker", None)
    thinker_model = getattr(thinker, "model", None)
    layers = getattr(thinker_model, "layers", None)
    if layers is None:
        raise RuntimeError("Model has no qwen.thinker.model.layers")
    candidate_indices: list[int] = []
    already_quantized: list[int] = []
    for layer_index, layer in enumerate(layers):
        experts = getattr(getattr(layer, "mlp", None), "experts", None)
        if isinstance(experts, FrozenInt8Experts):
            already_quantized.append(layer_index)
        elif (
            isinstance(experts, nn.Module)
            and hasattr(experts, "gate_up_proj")
            and hasattr(experts, "down_proj")
        ):
            candidate_indices.append(layer_index)
    if already_quantized:
        raise RuntimeError(
            "Thinker experts are already INT8-quantized in layers "
            f"{already_quantized}"
        )
    if len(candidate_indices) != options.expected_modules:
        raise RuntimeError(
            "Thinker routed-expert module inventory mismatch: "
            f"actual={len(candidate_indices)} expected={options.expected_modules}"
        )
    source_parameters = sum(
        int(parameter.numel())
        for layer_index in candidate_indices
        for parameter in layers[layer_index].mlp.experts.parameters()
    )
    if source_parameters != options.expected_parameters:
        raise RuntimeError(
            "Thinker routed-expert parameter inventory mismatch: "
            f"actual={source_parameters} expected={options.expected_parameters}"
        )
    source_storage_bytes = sum(
        int(parameter.numel()) * int(parameter.element_size())
        for layer_index in candidate_indices
        for parameter in layers[layer_index].mlp.experts.parameters()
    )
    module_names: list[str] = []
    quantized_storage_bytes = 0
    scale_elements = 0
    compute_dtype: torch.dtype | None = None
    for layer_index in candidate_indices:
        experts = layers[layer_index].mlp.experts
        replacement = FrozenInt8Experts.from_frozen(
            experts,
            expert_chunk_size=options.expert_chunk_size,
        )
        layers[layer_index].mlp.experts = replacement
        compute_dtype = replacement.compute_dtype
        module_names.append(
            f"qwen.thinker.model.layers.{layer_index}.mlp.experts"
        )
        quantized_storage_bytes += replacement.storage_bytes()
        scale_elements += int(replacement.gate_up_proj_scale.numel())
        scale_elements += int(replacement.down_proj_scale.numel())
        del experts
    remaining_source_parameters = sum(
        int(parameter.numel())
        for name, parameter in model.named_parameters()
        if name.startswith("qwen.thinker.model.layers.")
        and ".mlp.experts." in name
    )
    if remaining_source_parameters:
        raise RuntimeError(
            "BF16 Thinker routed-expert parameters remain after INT8 "
            f"conversion: {remaining_source_parameters}"
        )
    quantized_modules = sum(
        isinstance(getattr(getattr(layer, "mlp", None), "experts", None), FrozenInt8Experts)
        for layer in layers
    )
    if quantized_modules != options.expected_modules:
        raise RuntimeError(
            "Quantized Thinker routed-expert module inventory mismatch: "
            f"actual={quantized_modules} expected={options.expected_modules}"
        )
    return {
        "enabled": True,
        "implementation": IMPLEMENTATION,
        "quantization_scheme": options.quantization_scheme,
        "module_count": quantized_modules,
        "module_names": module_names,
        "source_parameters": source_parameters,
        "source_storage_bytes": source_storage_bytes,
        "source_storage_gib": source_storage_bytes / float(1 << 30),
        "quantized_storage_bytes": quantized_storage_bytes,
        "quantized_storage_gib": quantized_storage_bytes / float(1 << 30),
        "saved_storage_bytes": source_storage_bytes - quantized_storage_bytes,
        "saved_storage_gib": (
            source_storage_bytes - quantized_storage_bytes
        )
        / float(1 << 30),
        "scale_elements": scale_elements,
        "expert_chunk_size": options.expert_chunk_size,
        "compute_dtype": str(compute_dtype),
    }
