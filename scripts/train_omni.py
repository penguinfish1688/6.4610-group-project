"""D2-style Transformer-XL SFT: 90-s chunks, 90-s detached memory, four GPUs."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timedelta, timezone
import gc
import json
import os
from pathlib import Path
import random
import signal
import time
import traceback

import numpy as np
import torch
import torch.distributed as dist

from scripts.distributed import (
    shard_model,
    local_tensor,
    synchronize_and_clip,
    save_checkpoint,
    read_checkpoint,
    restore_checkpoint,
)
from scripts.xl_data import chunk_tensors, move_state
from scripts.loss_config import configure_loss, speech_eos_weight
from scripts.checkpoint_retention import checkpoint_retention_policy


def validate_resume_config(saved, current, step):
    """Allow run controls and an explicit speech-EOS loss override on resume."""
    controls = {"max_updates", "probe_every", "checkpoint_every"}
    saved = dict(
        saved,
        talker_codec_eos_weight=speech_eos_weight(saved),
        checkpoint_retention=checkpoint_retention_policy(saved),
    )
    current = dict(
        current,
        talker_codec_eos_weight=speech_eos_weight(current),
        checkpoint_retention=checkpoint_retention_policy(current),
    )
    changed = {key for key in saved.keys() | current.keys() if saved.get(key) != current.get(key)}
    unsupported = changed - controls - {"talker_codec_eos_weight", "checkpoint_retention"}
    if unsupported:
        raise ValueError(f"Resume training configuration changed: {sorted(unsupported)}")
    if current["max_updates"] < max(step, saved["max_updates"]):
        raise ValueError("Resume step limit must not decrease")
    if any(not isinstance(current[key], int) or current[key] < 1 for key in controls):
        raise ValueError("Run step limit and reporting intervals must be positive integers")
    return {
        key: {"previous": saved.get(key), "current": current.get(key)} for key in sorted(changed)
    }


def run(cfg):
    rank, local = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    if world != 4 or cfg["world_size"] != 4:
        raise ValueError("Expected one four-GPU node")
    torch.set_num_threads(cfg["cpu_threads_per_rank"])
    torch.set_num_interop_threads(1)
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    dist.init_process_group("nccl", timeout=timedelta(minutes=45), device_id=device)
    topology = [None] * world
    dist.all_gather_object(topology, (os.uname().nodename, local, torch.cuda.get_device_name()))
    if len({row[0] for row in topology}) != 1 or any("A6000" not in row[2] for row in topology):
        raise ValueError(f"Expected one A6000 node: {topology}")
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    output = Path(cfg["output"])
    output.mkdir(parents=True, exist_ok=True)
    logfile = (output / f"metrics-rank{rank}.jsonl").open("a", buffering=1)

    def log(event, **values):
        line = json.dumps(
            dict(
                event=event,
                time=time.time(),
                rank=rank,
                job=os.environ.get("SLURM_JOB_ID"),
                **values,
            ),
            default=str,
        )
        logfile.write(line + "\n")
        print(line, flush=True)

    def memory():
        return dict(
            allocated_gb=round(torch.cuda.memory_allocated() / 2**30, 3),
            peak_gb=round(torch.cuda.max_memory_allocated() / 2**30, 3),
        )

    log("starting", topology=topology, config=cfg, torch=torch.__version__)
    from scripts.int8_sharding import validate

    validate(device)
    log("quantized_storage_preflight_passed", raw_bytes_unchanged=True, gradient_parity=True)
    from scripts.causal_attention import validate as validate_causal_attention, set_causal_path

    log("causal_attention_preflight_passed", cases=validate_causal_attention(device))
    manifest = output / "data_manifest.json"
    if (output / "latest.json").exists() and manifest.exists():
        # Resume the exact prepared corpus recorded by the checkpoint's run.
        rows = json.loads(manifest.read_text())["rows"]
        if any(not Path(row["path"]).is_file() for row in rows):
            raise FileNotFoundError("A checkpoint's prepared conversation is missing")
        log("prepared_data_reused", conversations=len(rows))
    else:
        from scripts.prepare_native_data import prepare
        from scripts.prepare_avqa import ready, prepare_avqa

        rows, processor, vision = prepare(cfg, device, log)
        deadline = datetime.fromisoformat(cfg["avqa_wait_deadline_utc"].replace("Z", "+00:00"))
        while True:
            if ready(Path(cfg["avqa_raw_root"])):
                avqa_rows = prepare_avqa(cfg, processor, vision, device, log)
                if avqa_rows:
                    rows.extend(avqa_rows)
                    break
            remaining = (deadline - datetime.now(timezone.utc)).total_seconds()
            if remaining <= 0:
                log("avqa_deadline_reached", sources=["instructs2s_dynamic"])
                break
            log("waiting_for_avqa", remaining_seconds=round(remaining))
            time.sleep(min(30, remaining))
        del processor, vision
    gc.collect()
    torch.cuda.empty_cache()
    sources = sorted({row["source"] for row in rows})
    pattern = [kind for kind in cfg["source_pattern"] if kind in sources]
    training = {
        kind: [row for row in rows if row["split"] == "train" and row["source"] == kind]
        for kind in sources
    }
    probes = [row for row in rows if row["split"] == "probe"]
    if any(len(v) < world for v in training.values()) or len(probes) < world:
        raise ValueError("Too few complete source-disjoint >=720-second conversations")
    if any(row["ticks"] * 0.08 < cfg["minimum_conversation_seconds"] for row in rows):
        raise ValueError("A prepared conversation is shorter than 720 seconds")
    if rank == 0:
        (output / "data_manifest.json").write_text(
            json.dumps(
                dict(rows=rows, source_pattern=pattern, official_test_used_for_training=False),
                indent=2,
            )
            + "\n"
        )
    from d2_qwen.model.loading import load_model
    from scripts.optimizer import configure_optimizer

    log("loading_cpu_backbone")
    model, release = load_model(cfg["d2_source"], device="cpu", offline=True, training=True)
    model.enable_vision()
    log("loss_configured", **configure_loss(model, cfg))
    if cfg.get("frozen_experts_int8"):
        from scripts.quantize_experts import quantize

        log("experts_quantized", **quantize(model, cfg, device))
    named, replicated, blocks = shard_model(model, cfg, device)
    optimizer, scheduler = configure_optimizer(model)
    log(
        "model_sharded",
        trainable_parameters=sum(p.numel() for p in named.values()),
        blocks=len(blocks),
        local_parameter_bytes=sum(
            local_tensor(p).numel() * p.element_size() for p in model.parameters()
        ),
        tokens_per_tick=model.thinker_tokens_per_frame,
        **memory(),
    )
    loaded = {}

    def payload(row):
        if row["path"] not in loaded:
            loaded[row["path"]] = torch.load(row["path"], mmap=True, weights_only=False)
        return loaded[row["path"]]

    def cohort_length(row):
        length = torch.tensor(row["ticks"], device=device)
        dist.all_reduce(length, op=dist.ReduceOp.MAX)
        return (int(length) + cfg["max_frames"] - 1) // cfg["max_frames"]

    def forward(row, chunk, state):
        start = chunk * cfg["max_frames"]
        batch = {
            k: v.to(device)
            for k, v in chunk_tensors(payload(row), start, cfg["max_frames"], model.pad_id).items()
        }
        full_prefix = state is None or state.valid_memory_frames == cfg["memory_frames"]
        set_causal_path(model, bool(batch["frame_mask"].all()) and full_prefix)
        result, next_state = model(
            **batch, audio_start_frame=start, state=state, memory_frames=cfg["memory_frames"]
        )
        num = torch.stack(
            (result["loss_text_local_numerator"], result["loss_codec_local_numerator"])
        )
        den = torch.stack(
            (result["loss_text_local_denominator"], result["loss_codec_local_denominator"])
        ).float()
        dist.all_reduce(den)
        loss = (num / den.clamp_min(1)).sum() * world
        total = num.detach().float().clone()
        dist.all_reduce(total)
        if not bool(torch.isfinite(total).all()):
            raise FloatingPointError("Nonfinite loss")
        return loss, total, den, next_state

    step, cohort_index, chunk_index = 0, 0, 0
    cursors = {kind: 0 for kind in sources}
    active_row, active_chunks, state = None, 0, None
    history, probe_history = [], []
    checkpoint = output / "latest.json"
    if checkpoint.exists():
        saved = read_checkpoint(output)
        changes = validate_resume_config(saved["config"], cfg, saved["step"])
        if saved["sources"] != sources:
            raise ValueError("Resume source selection changed")
        if changes:
            log("resume_controls_changed", changes=changes)
        restore_checkpoint(saved, named, optimizer)
        scheduler.load_state_dict(saved["scheduler"])
        step, cohort_index, chunk_index = saved["step"], saved["cohort_index"], saved["chunk_index"]
        cursors = saved["cursors"]
        active_row, active_chunks = saved["active_row"], saved["active_chunks"]
        state = move_state(saved["xl_state"], device)
        history, probe_history = saved["history"], saved["probe_history"]
        torch.set_rng_state(saved["rng_torch"])
        torch.cuda.set_rng_state(saved["rng_cuda"], device)
        del saved
        gc.collect()
        log("resumed", update=step, cohort=cohort_index, chunk=chunk_index)

    @torch.no_grad()
    def probe():
        model.eval()
        totals, denominator = torch.zeros(2, device=device), torch.zeros(2, device=device)
        by_source = {}
        for start in range(0, len(probes) - world + 1, world):
            group_sources = {row["source"] for row in probes[start : start + world]}
            if len(group_sources) != 1:
                raise ValueError("Development probe groups must be homogeneous by source")
            kind = next(iter(group_sources))
            if kind not in by_source:
                by_source[kind] = [torch.zeros(2, device=device), torch.zeros(2, device=device)]
            row, probe_state = probes[start + rank], None
            chunks = min(cohort_length(row), cfg["probe_chunks"])
            for chunk in range(chunks):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    _, num, den, probe_state = forward(row, chunk, probe_state)
                totals += num
                denominator += den
                by_source[kind][0] += num
                by_source[kind][1] += den
            del probe_state
        values = (totals / denominator.clamp_min(1)).tolist()
        source_metrics = {}
        for kind, (numerator, count) in by_source.items():
            parts = (numerator / count.clamp_min(1)).tolist()
            source_metrics[kind] = dict(loss=sum(parts), text_loss=parts[0], codec_loss=parts[1])
        record = dict(
            update=step,
            loss=sum(values),
            text_loss=values[0],
            codec_loss=values[1],
            talker_codec_eos_weight=model.talker_codec_eos_weight,
            by_source=source_metrics,
        )
        probe_history.append(record)
        log("probe", **record, **memory())
        model.train()
        return record

    last_saved_step = None

    def save():
        nonlocal last_saved_step
        if last_saved_step == step:
            return
        save_checkpoint(
            output,
            named,
            optimizer,
            dict(
                step=step,
                config=cfg,
                release=release,
                sources=sources,
                cursors=cursors,
                cohort_index=cohort_index,
                chunk_index=chunk_index,
                active_row=active_row,
                active_chunks=active_chunks,
                xl_state=move_state(state, "cpu"),
                scheduler=scheduler.state_dict(),
                history=history,
                probe_history=probe_history,
                rng_torch=torch.get_rng_state(),
                rng_cuda=torch.cuda.get_rng_state(device),
            ),
        )
        log("checkpoint", update=step, cohort=cohort_index, chunk=chunk_index)
        last_saved_step = step

    stopping = False

    def stop(_signal, _frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGUSR1, stop)
    profile_update = step + 2 if os.environ.get("D2_OMNI_PROFILE_NEXT") == "1" else -1
    probe()
    while step < cfg["max_updates"]:
        kind = pattern[cohort_index % len(pattern)]
        if active_row is None:
            choices = training[kind]
            active_row = choices[(cursors[kind] * world + rank) % len(choices)]
            active_chunks = cohort_length(active_row)
            state, chunk_index = None, 0
            log(
                "cohort_start",
                source=kind,
                row=active_row["path"],
                chunks=active_chunks,
                duration_seconds=active_row["ticks"] * 0.08,
            )
        started = time.monotonic()
        profiler = (
            torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
            )
            if rank == 0 and step + 1 == profile_update
            else None
        )
        markers = [torch.cuda.Event(enable_timing=True) for _ in range(5)]
        with profiler if profiler is not None else nullcontext():
            optimizer.zero_grad(set_to_none=True)
            markers[0].record()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, num, den, next_state = forward(active_row, chunk_index, state)
            markers[1].record()
            loss.backward()
            markers[2].record()
            norm = synchronize_and_clip(named, replicated, 1.0)
            markers[3].record()
            optimizer.step()
            scheduler.step()
            markers[4].record()
            markers[4].synchronize()
        state = next_state
        step += 1
        chunk_index += 1
        values = (num / den.clamp_min(1)).tolist()
        record = dict(
            update=step,
            source=kind,
            cohort=cohort_index,
            chunk=chunk_index,
            loss=sum(values),
            text_loss=values[0],
            codec_loss=values[1],
            grad_norm=float(norm),
            talker_codec_eos_weight=model.talker_codec_eos_weight,
            seconds=time.monotonic() - started,
            lr=optimizer.param_groups[0]["lr"],
            thinker_cache_tokens=state.thinker_cache.get_seq_length(),
            memory_detached=not state.thinker_cache.layers[0].keys.requires_grad,
        )
        record["fast_causal_attention"] = model._d2_using_fast_causal
        history.append(record)
        record.update(
            {
                name + "_seconds": markers[i].elapsed_time(markers[i + 1]) / 1000
                for i, name in enumerate(("forward", "backward", "clip", "optimizer"))
            }
        )
        log("update", **record, **memory())
        if profiler is not None:
            profiler.export_chrome_trace(str(output / f"profile-step-{step}.json"))
            (output / f"profile-step-{step}.txt").write_text(
                profiler.key_averages().table(sort_by="self_cuda_time_total", row_limit=40)
            )
            log("cuda_profile", update=step, path=str(output / f"profile-step-{step}.json"))
        if chunk_index == active_chunks:
            cursors[kind] += 1
            cohort_index += 1
            active_row, active_chunks, state, chunk_index = None, 0, None, 0
        if step % cfg["probe_every"] == 0:
            probe()
        stop_tensor = torch.tensor(int(stopping), device=device)
        dist.all_reduce(stop_tensor, op=dist.ReduceOp.MAX)
        if step == 4 or step % cfg["checkpoint_every"] == 0 or bool(stop_tensor):
            save()
        if step == 4:
            before = probe()
            saved = read_checkpoint(output)
            restore_checkpoint(saved, named, optimizer)
            state = move_state(saved["xl_state"], device)
            del saved
            gc.collect()
            after = probe()
            if abs(before["loss"] - after["loss"]) > 1e-5:
                raise RuntimeError("Checkpoint reload changed fixed validation loss")
            log("checkpoint_reload_validated", update=step, recurrence_restored=True)
        if bool(stop_tensor):
            break
    save()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train-sls.json", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    try:
        run(config)
    except BaseException:
        # Keep an independent traceback if torchrun/Slurm stderr is interrupted.
        try:
            failure = Path(config["output"]) / (
                f"crash-{os.environ.get('SLURM_JOB_ID', 'local')}-"
                f"rank{os.environ.get('RANK', 'unknown')}.txt"
            )
            failure.write_text(traceback.format_exc())
        except OSError:
            pass
        raise
