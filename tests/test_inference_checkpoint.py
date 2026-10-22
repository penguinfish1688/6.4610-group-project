"""Validate adapter assembly before free-running inference."""

import pytest
import torch

from scripts.inference_checkpoint import merge_parameter


def test_uneven_shards_restore_exact_global_parameter():
    expected = torch.arange(15).reshape(5, 3).float()
    pieces = list(torch.tensor_split(expected, 4, dim=0))
    torch.testing.assert_close(merge_parameter(pieces, expected.shape), expected)


def test_single_row_with_empty_shards():
    expected = torch.ones(1, 7)
    pieces = list(torch.tensor_split(expected, 4, dim=0))
    torch.testing.assert_close(merge_parameter(pieces, expected.shape), expected)


def test_replicated_parameter_must_agree_on_every_rank():
    value = torch.randn(8)
    pieces = [value.clone() for _ in range(4)]
    torch.testing.assert_close(merge_parameter(pieces, value.shape), value)
    pieces[-1][0] += 1
    with pytest.raises(ValueError, match="disagree"):
        merge_parameter(pieces, value.shape)


def test_invalid_shard_layout_is_rejected():
    with pytest.raises(ValueError, match="dimensions"):
        merge_parameter([torch.ones(2, 3)] * 4, (8, 4))
    with pytest.raises(ValueError, match="shape"):
        merge_parameter([torch.ones(2, 3)] * 4, (7, 3))
