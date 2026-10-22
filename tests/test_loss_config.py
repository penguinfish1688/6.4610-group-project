"""Speech EOS changes must reach the loss without changing Thinker controls."""

from types import SimpleNamespace

import pytest
import torch

from d2_qwen.model.base import masked_ce, two_signal_codec_supervision
from d2_qwen.model.constants import CODEC_SILENCE_FRAME
from scripts.loss_config import configure_loss, speech_eos_weight


def test_speech_eos_override_keeps_text_control_weights():
    model = SimpleNamespace(
        talker_codec_eos_weight=20.0, assistant_start_weight=20.0, assistant_end_weight=20.0
    )
    result = configure_loss(model, dict(talker_codec_eos_weight=5.0))
    assert result == dict(
        talker_codec_eos_weight=5.0, text_response_weight=20.0, text_interrupt_weight=20.0
    )
    assert speech_eos_weight({}) == 20.0


def test_eos_mask_and_actual_gradient_have_five_to_one_weight():
    codec = torch.tensor(CODEC_SILENCE_FRAME).view(1, 1, 16).repeat(1, 3, 1)
    codec[0, 0, 0], codec[0, 1, 0], codec[0, 2, 0] = 2149, 0, 2150
    valid = torch.ones(1, 3, dtype=torch.bool)
    effective, mask, residual, weight = two_signal_codec_supervision(
        codec,
        valid,
        valid,
        codec[:, :, 0] == 2149,
        codec[:, :, 0] == 2150,
        torch.tensor(CODEC_SILENCE_FRAME),
        codec_eos_weight=5.0,
    )
    assert not bool(mask[0, 0]) and bool(mask[0, 2]) and not bool(residual[0, 2])
    assert int(effective[0, 2, 0]) == 2150
    assert weight.tolist() == [[1.0, 1.0, 5.0]]
    logits = torch.zeros(1, 3, 3072, requires_grad=True)
    masked_ce(logits, effective[:, :, 0], mask, frame_weight=weight).backward()
    assert torch.count_nonzero(logits.grad[0, 0]) == 0
    torch.testing.assert_close(logits.grad[0, 2, 2150] / logits.grad[0, 1, 0], torch.tensor(5.0))


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf"), True, "5"])
def test_reject_invalid_eos_weights(value):
    with pytest.raises(ValueError, match="finite positive"):
        speech_eos_weight(dict(talker_codec_eos_weight=value))
