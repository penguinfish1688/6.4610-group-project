import torch
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeRMSNorm

from scripts.precision import preserve_norm_dtype


def test_fp32_norm_master_keeps_bf16_activations_and_gradients():
    norm = Qwen3OmniMoeRMSNorm(16).float()
    preserve_norm_dtype(norm)
    value = torch.randn(2, 5, 16, dtype=torch.bfloat16, requires_grad=True)
    output = norm(value)
    assert output.dtype == torch.bfloat16
    output.float().square().sum().backward()
    assert norm.weight.grad.dtype == torch.float32
    assert bool(torch.isfinite(norm.weight.grad).all())
