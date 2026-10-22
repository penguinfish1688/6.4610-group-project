"""Explicit run-level overrides of the native D2 speech EOS loss."""

import math
from numbers import Real


def speech_eos_weight(config):
    value = config.get("talker_codec_eos_weight", 20.0)
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError("talker_codec_eos_weight must be a finite positive number")
    return float(value)


def configure_loss(model, config):
    model.talker_codec_eos_weight = speech_eos_weight(config)
    return dict(
        talker_codec_eos_weight=model.talker_codec_eos_weight,
        text_response_weight=model.assistant_start_weight,
        text_interrupt_weight=model.assistant_end_weight,
    )
