"""Extending a run must preserve its training and recurrence recipe."""

import pytest

from scripts.train_omni import validate_resume_config


@pytest.fixture
def config():
    return dict(
        max_updates=1000,
        probe_every=10,
        checkpoint_every=25,
        memory_frames=1125,
        source_pattern=["instructs2s_dynamic", "avqa"],
        vision=dict(tokens_per_image=12, fps=1),
    )


def test_extend_completed_run(config):
    current = dict(config, max_updates=10000, probe_every=100, checkpoint_every=250)
    changes = validate_resume_config(config, current, step=1000)
    assert changes["max_updates"] == dict(previous=1000, current=10000)
    assert len(changes) == 3
    assert validate_resume_config(current, current, step=5000) == {}


def test_explicit_speech_eos_override_preserves_legacy_default(config):
    current = dict(config, max_updates=10000, talker_codec_eos_weight=5.0)
    changes = validate_resume_config(config, current, step=1000)
    assert changes["talker_codec_eos_weight"] == dict(previous=20.0, current=5.0)
    assert validate_resume_config(config, dict(config, talker_codec_eos_weight=20.0), 1000) == {}
    assert validate_resume_config(current, current, 1250) == {}


def test_resume_still_rejects_other_loss_changes(config):
    with pytest.raises(ValueError, match="training configuration changed"):
        validate_resume_config(config, dict(config, assistant_end_weight=5.0), step=1000)


def test_resume_allows_latest_only_checkpoint_retention(config):
    current = dict(config, checkpoint_retention="latest_only")
    changes = validate_resume_config(config, current, step=1000)
    assert changes == dict(checkpoint_retention=dict(previous="all", current="latest_only"))
    assert validate_resume_config(current, current, step=1000) == {}


def test_resume_rejects_unknown_checkpoint_retention(config):
    with pytest.raises(ValueError, match="checkpoint_retention"):
        validate_resume_config(config, dict(config, checkpoint_retention="invalid"), step=1000)


@pytest.mark.parametrize(
    "change",
    [
        dict(memory_frames=100),
        dict(source_pattern=["avqa"]),
        dict(vision=dict(tokens_per_image=24, fps=1)),
    ],
)
def test_reject_training_recipe_changes(config, change):
    with pytest.raises(ValueError, match="training configuration changed"):
        validate_resume_config(config, dict(config, **change), step=1000)


def test_reject_shorter_limit(config):
    with pytest.raises(ValueError, match="must not decrease"):
        validate_resume_config(config, dict(config, max_updates=999), step=1000)


@pytest.mark.parametrize(
    "change", [dict(probe_every=0), dict(checkpoint_every=-1), dict(max_updates=10000.5)]
)
def test_reject_invalid_run_controls(config, change):
    with pytest.raises(ValueError, match="positive integers"):
        validate_resume_config(config, dict(config, **change), step=1000)
