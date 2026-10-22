"""Bit-preserving floating storage for frozen INT8 weights under FSDP2."""

import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.utils.checkpoint import checkpoint


def floating_storage(tensor):
    if tensor.dtype != torch.int8 or tensor.shape[-1] % 2:
        raise ValueError("Expected contiguous INT8 weights with an even last dimension")
    return torch.nn.Parameter(tensor.contiguous().view(torch.bfloat16), requires_grad=False)


def validate(device):
    """Check raw-byte all-gather and backward replay before expensive loading."""

    class Layer(torch.nn.Module):
        def __init__(self, raw, sharded):
            super().__init__()
            self.weight = (
                floating_storage(raw) if sharded else torch.nn.Parameter(raw, requires_grad=False)
            )
            self.adapter = torch.nn.Parameter(torch.zeros(32, 32, device=device))

        def compute(self, value):
            weight = self.weight.view(torch.int8).to(value.dtype)
            return value @ weight.T + torch.nn.functional.linear(
                value, self.adapter.to(value.dtype)
            )

        def forward(self, value):
            return checkpoint(self.compute, value, use_reentrant=False)

    raw = torch.arange(1024, device=device).to(torch.int8).reshape(32, 32)
    reference, candidate = Layer(raw.clone(), False), Layer(raw.clone(), True)
    mesh = init_device_mesh("cuda", (torch.distributed.get_world_size(),))
    fully_shard(
        candidate,
        mesh=mesh,
        mp_policy=MixedPrecisionPolicy(param_dtype=None, reduce_dtype=torch.float32),
    )
    left = (
        torch.linspace(-1, 1, 64, device=device, dtype=torch.bfloat16)
        .reshape(2, 32)
        .requires_grad_()
    )
    right = left.detach().clone().requires_grad_()
    expected, actual = reference(left), candidate(right)
    expected.float().square().mean().backward()
    actual.float().square().mean().backward()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(right.grad, left.grad, rtol=0, atol=0)
    torch.testing.assert_close(
        candidate.adapter.grad.full_tensor(), reference.adapter.grad, rtol=0, atol=0
    )
    if not torch.equal(candidate.weight.full_tensor().view(torch.int8), raw):
        raise RuntimeError("FSDP changed frozen INT8 payload bytes")
    del candidate, reference
    torch.cuda.empty_cache()
