"""Native D2 INT8 inference routing, with no external SLS source dependencies."""

import torch
from model import frozen_moe_int8 as kernels

_REFERENCE_FROZEN_INT8_GROUPED_LINEAR = kernels._frozen_int8_grouped_linear

def _use_cuda_fallback_routed_bmm(
    *,
    device_type: str,
    device_capability: tuple[int, int] | None,
    compute_dtype: torch.dtype,
    grad_enabled: bool,
) -> bool:
    """Replace Torch 2.11's host-loop grouped_mm fallback during inference."""

    return (
        device_type == "cuda"
        and device_capability is not None
        and not (
            int(device_capability[0]) in (9, 10)
            and compute_dtype == torch.bfloat16
        )
        and not grad_enabled
    )

def _fixed_shape_routed_int8_linear(
    inputs: torch.Tensor,
    quantized: torch.Tensor,
    scales: torch.Tensor,
    offsets: torch.Tensor,
    *,
    compute_dtype: torch.dtype,
) -> torch.Tensor:
    """Apply each route's expert without grouped_mm's host-offset fallback.

    ``offsets`` contains cumulative route counts in expert order.  Comparing a
    fixed route index vector against those offsets reconstructs the expert ID
    entirely on device.  The resulting gather and BMM have shapes determined
    only by the fixed input shape, so changing router decisions remains safe
    across CUDA-graph replays.
    """

    route_count = int(inputs.shape[0])
    expert_count = int(quantized.shape[0])
    if expert_count < 1:
        raise ValueError("routed INT8 linear requires at least one expert")
    if tuple(scales.shape) != (*quantized.shape[:-1], 1):
        raise ValueError(
            "routed INT8 scale shape mismatch: "
            f"quantized={tuple(quantized.shape)} scales={tuple(scales.shape)}"
        )
    if offsets.ndim != 1 or int(offsets.numel()) != expert_count:
        raise ValueError(
            "routed INT8 offsets must contain one entry per expert: "
            f"offsets={tuple(offsets.shape)} experts={expert_count}"
        )
    if offsets.dtype != torch.int32:
        raise TypeError(f"routed INT8 offsets must be int32, got {offsets.dtype}")

    route_indices = torch.arange(
        route_count,
        device=offsets.device,
        dtype=offsets.dtype,
    )
    expert_indices = torch.sum(
        route_indices.unsqueeze(1) >= offsets.unsqueeze(0),
        dim=1,
    ).clamp_(max=expert_count - 1)
    routed_quantized = quantized.index_select(0, expert_indices)
    routed_scales = scales.index_select(0, expert_indices)
    routed_weight = routed_quantized.to(dtype=compute_dtype)
    routed_weight.mul_(routed_scales.to(dtype=compute_dtype))
    return torch.bmm(
        inputs.to(dtype=compute_dtype).unsqueeze(1),
        routed_weight.transpose(-2, -1),
    ).squeeze(1)

def _d2_frozen_int8_grouped_linear(
    inputs: torch.Tensor,
    quantized: torch.Tensor,
    scales: torch.Tensor,
    offsets: torch.Tensor,
    *,
    compute_dtype: torch.dtype,
) -> torch.Tensor:
    capability = (
        torch.cuda.get_device_capability(inputs.device)
        if inputs.device.type == "cuda"
        else None
    )
    if _use_cuda_fallback_routed_bmm(
        device_type=inputs.device.type,
        device_capability=capability,
        compute_dtype=compute_dtype,
        grad_enabled=torch.is_grad_enabled(),
    ):
        return _fixed_shape_routed_int8_linear(
            inputs,
            quantized,
            scales,
            offsets,
            compute_dtype=compute_dtype,
        )
    return _REFERENCE_FROZEN_INT8_GROUPED_LINEAR(
        inputs,
        quantized,
        scales,
        offsets,
        compute_dtype=compute_dtype,
    )

kernels._frozen_int8_grouped_linear = _d2_frozen_int8_grouped_linear
