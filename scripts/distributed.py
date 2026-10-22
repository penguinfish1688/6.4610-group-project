"""Node-local FSDP2 sharding with FP32 adapters and replicated audio frontends."""

import hashlib
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.tensor import DTensor

from scripts.checkpoint_retention import checkpoint_retention_policy, prune_checkpoints


def local_tensor(value):
    return value.to_local() if isinstance(value, DTensor) else value


def shard_model(model, config, device):
    """Shard each decoder before GPU placement; never place the full model there."""
    # Some native adapters inherit BF16 from their source projections. Keep
    # master weights/Adam state in FP32 while autocast handles compute.
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.float()
    shards = config["gpus_per_node"]
    world = dist.get_world_size()
    if world % shards:
        raise ValueError("World size must be divisible by the node-local shard count")
    mesh = init_device_mesh(
        device.type, (world // shards, shards), mesh_dim_names=("replicate", "shard")
    )
    policy = MixedPrecisionPolicy(
        param_dtype=None, reduce_dtype=torch.float32, cast_forward_inputs=False
    )
    blocks = []
    # The small code predictor and frontends remain replicated. Decoder layers
    # account for the large backbone, including all routed MoE experts.
    for transformer in (model.qwen.thinker.model, model.qwen.talker.model):
        layers = list(transformer.layers)
        for layer in layers:
            fully_shard(layer, mesh=mesh, mp_policy=policy, reshard_after_forward=True)
            blocks.append(layer)
        for i, layer in enumerate(layers):
            if i + 1 < len(layers):
                layer.set_modules_to_forward_prefetch([layers[i + 1]])
            if i:
                layer.set_modules_to_backward_prefetch([layers[i - 1]])
    replicated = {p for p in model.parameters() if not isinstance(p, DTensor)}
    # Move only unsharded parameters, preserving shared parameter identities.
    with torch.no_grad():
        for parameter in replicated:
            parameter.data = parameter.data.to(device)
        for module in model.modules():
            for name, buffer in module._buffers.items():
                if buffer is not None:
                    module._buffers[name] = buffer.to(device)
    # A parameterless root coordinates the child FSDP states. Audio preprocessing
    # and embeddings can still be called outside its forward without collectives.
    fully_shard(model, mesh=mesh, mp_policy=policy, ignored_params=replicated)
    trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
    if any(p.dtype != torch.float32 for p in trainable.values()):
        raise RuntimeError("Every trainable parameter must have FP32 storage")
    replicated_trainable = [p for p in trainable.values() if not isinstance(p, DTensor)]
    for parameter in replicated_trainable:
        dist.broadcast(parameter.data, src=0)
    return trainable, replicated_trainable, blocks


def synchronize_and_clip(named, replicated, max_norm):
    """FSDP already averaged sharded grads; reduce only replicated lane grads."""
    world = dist.get_world_size()
    # Boundary-only rows may be unused in continuation chunks. Preserve Adam's
    # native None-gradient behavior when a parameter is unused on every rank.
    used = torch.tensor(
        [p.grad is not None for p in named.values()],
        dtype=torch.int32,
        device=next(iter(named.values())).device,
    )
    dist.all_reduce(used, op=dist.ReduceOp.MAX)
    for parameter, active in zip(named.values(), used.tolist(), strict=True):
        if active and parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
    for parameter in replicated:
        if parameter.grad is None:
            continue
        dist.all_reduce(parameter.grad)
        parameter.grad.div_(world)
    device = next(iter(named.values())).device
    squared = torch.zeros((), dtype=torch.float64, device=device)
    missing = []
    for name, parameter in named.items():
        if parameter.grad is None:
            missing.append(name)
            continue
        grad = local_tensor(parameter.grad)
        if grad.dtype != torch.float32:
            raise RuntimeError(f"Non-FP32 adapter gradient: {name}")
        # Count each global element once, including replicated mesh dimensions.
        copies = world
        if isinstance(parameter, DTensor):
            copies = world // parameter.device_mesh.size(1)
        squared += grad.double().square().sum() / copies
    dist.all_reduce(squared)
    norm = squared.sqrt().float()
    if not bool(torch.isfinite(norm)):
        raise FloatingPointError(f"Nonfinite gradient norm: {norm}")
    coefficient = (max_norm / (norm + 1e-6)).clamp(max=1.0)
    for parameter in named.values():
        if parameter.grad is not None:
            local_tensor(parameter.grad).mul_(coefficient)
    return norm


def _cpu(value):
    if torch.is_tensor(value):
        return local_tensor(value).detach().cpu()
    if isinstance(value, dict):
        return {k: _cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_cpu(v) for v in value]
    return value


def _sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def save_checkpoint(output, named, optimizer, payload):
    """Publish an atomic manifest only after every rank's shard is durable."""
    output = Path(output)
    retention = checkpoint_retention_policy(payload.get("config", {}))
    rank = dist.get_rank()
    generation = f"step-{payload['step']:08d}-{os.environ.get('SLURM_JOB_ID', 'local')}"
    folder = output / generation
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"rank-{rank:03d}.pt"
    temporary = path.with_suffix(".partial")
    states = {n: _cpu(optimizer.state[p]) for n, p in named.items()}
    torch.save(
        dict(
            **payload,
            weights={n: _cpu(p) for n, p in named.items()},
            optimizer_states=states,
            optimizer_groups=[
                {k: v for k, v in g.items() if k != "params"} for g in optimizer.param_groups
            ],
        ),
        temporary,
    )
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    temporary.replace(path)
    _sync_directory(folder)
    receipt = dict(path=str(path.relative_to(output)), sha256=digest, bytes=path.stat().st_size)
    receipts = [None] * dist.get_world_size()
    dist.all_gather_object(receipts, receipt)
    manifest = dict(step=payload["step"], shards=receipts, world_size=dist.get_world_size())
    if rank == 0:
        pending = output / "latest.partial"
        pending.write_text(json.dumps(manifest, indent=2) + "\n")
        with pending.open("rb") as stream:
            os.fsync(stream.fileno())
        pending.replace(output / "latest.json")
        _sync_directory(output)
    dist.barrier()
    if retention == "latest_only":
        cleanup = [None]
        if rank == 0:
            try:
                report = prune_checkpoints(output, manifest)
                pending = output / "retention-latest.partial"
                pending.write_text(json.dumps(report, indent=2) + "\n")
                pending.replace(output / "retention-latest.json")
                cleanup[0] = dict(ok=True)
            except Exception as error:
                cleanup[0] = dict(ok=False, error=f"{type(error).__name__}: {error}")
        dist.broadcast_object_list(cleanup, src=0)
        if not cleanup[0]["ok"]:
            raise RuntimeError(f"Checkpoint saved but retention failed: {cleanup[0]['error']}")
    return manifest


def read_checkpoint(output):
    output = Path(output)
    manifest = json.loads((output / "latest.json").read_text())
    if manifest["world_size"] != dist.get_world_size():
        raise ValueError("Resume requires the original distributed topology")
    entry = manifest["shards"][dist.get_rank()]
    path = output / entry["path"]
    with path.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != entry["sha256"]:
            raise RuntimeError(f"Checkpoint checksum mismatch: {path}")
    return torch.load(path, map_location="cpu", weights_only=False, mmap=True)


def restore_checkpoint(payload, named, optimizer):
    if set(payload["weights"]) != set(named):
        raise ValueError("Checkpoint trainable inventory changed")
    with torch.no_grad():
        for name, parameter in named.items():
            local_tensor(parameter).copy_(payload["weights"][name])
            state = {}
            for key, value in payload["optimizer_states"][name].items():
                if torch.is_tensor(value):
                    # Loaded optimizer tensors can alias a memory-mapped shard.
                    # CPU step counters must also release that file for NFS pruning.
                    value = (
                        value.clone() if key == "step" else value.to(parameter.device, copy=True)
                    )
                    if key != "step" and isinstance(parameter, DTensor):
                        value = DTensor.from_local(
                            value,
                            parameter.device_mesh,
                            parameter.placements,
                            shape=parameter.shape,
                            stride=parameter.stride(),
                        )
                state[key] = value
            optimizer.state[parameter] = state
    for group, saved in zip(optimizer.param_groups, payload["optimizer_groups"], strict=True):
        group.update(saved)
