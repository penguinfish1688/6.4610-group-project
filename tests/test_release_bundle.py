"""Check release portability, integrity and the native INT8 inference fallback."""

import hashlib
import json

import pytest
import torch
from safetensors.torch import save_file

from d2.hub import asset
from scripts.inference_checkpoint import restore_inference_checkpoint
from model.int8_runtime import _fixed_shape_routed_int8_linear


def test_local_base_asset_survives_bundle_relocation(tmp_path):
    release = tmp_path / "arbitrary-location" / "base_d2"
    base = release.parent / "base_model"
    release.mkdir(parents=True)
    base.mkdir()
    config = {"base_model": {"repo_id": "Qwen/model", "local_path": "../base_model"}}
    assert asset(config, "base_model", root=release) == base
    with pytest.raises(ValueError, match="root"):
        asset(config, "base_model")
    config["base_model"]["local_path"] = "../../outside"
    with pytest.raises(ValueError, match="escapes"):
        asset(config, "base_model", root=release)


def test_safe_release_restores_and_rejects_corruption(tmp_path):
    model = torch.nn.Linear(3, 2)
    weights = {name: torch.randn_like(p) for name, p in model.named_parameters()}
    path = tmp_path / "model.safetensors"
    save_file(weights, str(path))
    manifest = dict(
        format="d2_omni.inference.v1", step=9000,
        weights=dict(path=path.name, bytes=path.stat().st_size,
                     sha256=hashlib.sha256(path.read_bytes()).hexdigest()),
    )
    (tmp_path / "latest.json").write_text(json.dumps(manifest))
    receipt = restore_inference_checkpoint(model, tmp_path)
    assert receipt["step"] == 9000 and receipt["parameters_restored"] == 2
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter, weights[name], rtol=0, atol=0)
    data = bytearray(path.read_bytes())
    data[-1] ^= 1
    path.write_bytes(data)
    with pytest.raises(ValueError, match="checksum"):
        restore_inference_checkpoint(model, tmp_path)


def test_int8_fallback_handles_empty_experts_and_matches_dense_routes():
    torch.manual_seed(17)
    quantized = torch.randint(-127, 128, (5, 7, 4), dtype=torch.int8)
    scales = torch.rand(5, 7, 1) / 127
    counts = torch.tensor([2, 0, 1, 0, 3], dtype=torch.int32)
    offsets = counts.cumsum(0).to(torch.int32)
    inputs = torch.randn(int(counts.sum()), 4)
    actual = _fixed_shape_routed_int8_linear(
        inputs, quantized, scales, offsets, compute_dtype=torch.float32,
    )
    dense = quantized.float() * scales
    expected = torch.cat([
        x @ weight.T for x, weight in zip(inputs.split(counts.tolist()), dense)
    ])
    torch.testing.assert_close(actual, expected)
