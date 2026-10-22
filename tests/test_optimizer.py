"""Mixed replicated/sharded Adam updates must match ordinary tensor Adam."""

import tempfile
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor, Shard

from scripts.optimizer import homogeneous_groups


def test_mixed_distributed_adam_matches_reference():
    with tempfile.TemporaryDirectory() as folder:
        dist.init_process_group(
            "gloo", init_method=f"file://{Path(folder) / 'store'}", rank=0, world_size=1
        )
        try:
            mesh = init_device_mesh("cpu", (1,))
            first = torch.nn.Parameter(torch.arange(6).float().reshape(2, 3))
            second = torch.nn.Parameter(distribute_tensor(first.detach() + 2, mesh, [Shard(0)]))
            expected = [
                torch.nn.Parameter(first.detach().clone()),
                torch.nn.Parameter(second.to_local().detach().clone()),
            ]
            groups = homogeneous_groups(
                [dict(name="native_group", params=[first, second], lr=0.001)]
            )
            actual_optimizer = torch.optim.AdamW(groups, foreach=True, weight_decay=0.0)
            reference_optimizer = torch.optim.AdamW(
                expected, lr=0.001, foreach=True, weight_decay=0.0
            )
            for step in range(1, 4):
                for parameter in [first, second] + expected:
                    parameter.grad = torch.full_like(parameter, step * 0.7)
                actual_optimizer.step()
                reference_optimizer.step()
            torch.testing.assert_close(first, expected[0])
            torch.testing.assert_close(second.to_local(), expected[1])
        finally:
            dist.destroy_process_group()
