"""Audit cached speech boundaries and the native D2 EOS loss weighting."""

import argparse
import ast
from collections import Counter
import inspect
import json
from pathlib import Path
import sys

import torch

from d2_qwen.model.base import (
    masked_ce,
    talker_state_mask_with_initial,
    two_signal_codec_supervision,
)
from d2_qwen.model.constants import CODEC_SILENCE_FRAME
from scripts.loss_config import speech_eos_weight


def native_helper_parity(cfg):
    root = Path(cfg["legacy_d2_root"])
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "qwen_duplex"))
    import duplex.model as original_base
    import duplex_async.model as original
    import d2_qwen.model.base as current

    result = {}
    for name in ("masked_ce", "two_signal_codec_supervision", "talker_state_mask_with_initial"):
        reference = getattr(original, name, None) or getattr(original_base, name)
        reference_ast = ast.dump(ast.parse(inspect.getsource(reference)), include_attributes=False)
        current_ast = ast.dump(
            ast.parse(inspect.getsource(getattr(current, name))), include_attributes=False
        )
        result[name] = reference_ast == current_ast
        assert result[name], f"Native D2 helper differs: {name}"
    return result


def run(args):
    torch.set_num_threads(2)
    cfg = json.loads(args.config.read_text())
    parity = native_helper_parity(cfg)
    output = Path(cfg["output"])
    manifest = json.loads((output / "data_manifest.json").read_text())
    recipe = json.loads((Path(cfg["prepared_root"]) / "audio_recipe.json").read_text())
    native_weight = recipe["native_config"]["loss"]["talker_codec_eos_weight"]
    weight = speech_eos_weight(cfg)
    eos_id, bos_id = 2150, 2149
    silence = torch.tensor(CODEC_SILENCE_FRAME)
    counts, gaps = Counter(), Counter()
    for row in manifest["rows"]:
        payload = torch.load(row["path"], mmap=True, weights_only=False)
        t = payload["tensors"]
        codec0 = t["codec_target"][:, :, 0]
        bos, eos = codec0 == bos_id, codec0 == eos_id
        valid, assistant = t["frame_mask"].bool(), t["assistant_mask"].bool()
        assert torch.equal(bos, t["text_target"] == 151669), row["path"]
        talker, tail_open = talker_state_mask_with_initial(
            codec0, codec_bos_id=bos_id, codec_eos_id=eos_id
        )
        assert not bool((assistant & ~talker).any()), row["path"]
        assert not bool((assistant & (bos | eos)).any()), row["path"]
        effective, c0_mask, residual_mask, frame_weight = two_signal_codec_supervision(
            t["codec_target"], valid, talker, bos, eos, silence, codec_eos_weight=weight
        )
        assert bool((effective[:, :, 0][eos] == eos_id).all()), row["path"]
        assert bool(c0_mask[eos].all()) and not bool(residual_mask[eos].any()), row["path"]
        assert bool((frame_weight[eos] == weight).all()), row["path"]
        assert bool((frame_weight[valid & ~eos] == 1).all()), row["path"]
        starts = bos[0].nonzero().flatten().tolist()
        for end in eos[0].nonzero().flatten().tolist():
            start = max(i for i in starts if i < end)
            audio = assistant[0, start:end].nonzero().flatten() + start
            if audio.numel():
                gap = end - int(audio[-1])
                gaps[gap] += 1
                assert gap == 1, (row["path"], start, end, int(audio[-1]))
            else:
                counts["empty_speech_episodes"] += 1
        counts["conversations"] += 1
        counts[f"{row['source']}/{row['split']}"] += 1
        counts["codec_bos_targets"] += int(bos.sum())
        counts["codec_eos_targets"] += int(eos.sum())
        counts["open_tails"] += int(tail_open.sum())
    # Check the actual CE gradient, not just the configured weight constant.
    logits = torch.zeros(1, 2, 3072, requires_grad=True)
    targets = torch.tensor([[0, eos_id]])
    loss = masked_ce(
        logits,
        targets,
        torch.ones(1, 2, dtype=torch.bool),
        frame_weight=torch.tensor([[1.0, weight]]),
    )
    loss.backward()
    ratio = float(logits.grad[0, 1, eos_id] / logits.grad[0, 0, 0])
    assert abs(ratio - weight) < 1e-4, ratio
    # Compare every independent AVQA episode to its immutable V3 row, translated
    # from the published +8 timeline to the requested +6 as in preparation.
    sys.path.append(cfg["shared_lskv_root"])
    from lskv.native_timing import retime_avqa_row

    view = Path(cfg["prepared_root"]) / "shared-avqa-metadata/sft_async_v3"
    source = json.loads((view / "manifest.json").read_text())
    by_index = {row["row_index"]: row for row in source["rows"]}
    for path in sorted(Path(cfg["prepared_root"]).glob("avqa-*.pt")):
        if "-xl-" in path.name:
            continue
        index = int(path.stem.rsplit("-", 1)[1])
        metadata = json.loads((view / by_index[index]["row_path"]).read_text())
        timeline = retime_avqa_row(metadata, cfg["control_offset_frames"])["timeline"]
        payload = torch.load(path, mmap=True, weights_only=False)
        t = payload["tensors"]
        actual_bos = (t["codec_target"][0, :, 0] == bos_id).nonzero().flatten().tolist()
        actual_eos = (t["codec_target"][0, :, 0] == eos_id).nonzero().flatten().tolist()
        assert actual_bos == [timeline["response_frame"]], path
        assert actual_eos == [timeline["talker_eos_frame"]], path
        assert timeline["talker_eos_frame"] == timeline["answer_last_audible_frame"] + 1, path
        counts["avqa_independent_episodes_compared_to_source"] += 1
    result = dict(
        passed=True,
        counts=dict(counts),
        speech_eos_after_last_audio_frame_histogram=dict(gaps),
        codec_eos_id=eos_id,
        codec_eos_cb0_weight=weight,
        native_d2_codec_eos_cb0_weight=native_weight,
        measured_eos_to_ordinary_cb0_gradient_ratio=ratio,
        native_d2_helpers_identical=parity,
        eos_in_cb0_loss=True,
        eos_excluded_from_residual_codebook_loss=True,
        avqa_source_timing_compared=True,
        shared_sources_modified=False,
    )
    (output / f"speech_eos_audit-eos{weight:g}.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/train-sls.json"))
    run(parser.parse_args())
