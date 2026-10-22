"""Native D2 Adam settings with homogeneous distributed foreach groups."""

import json
from pathlib import Path

import torch
from torch.distributed.tensor import DTensor

from d2.training import scheduler
from d2_qwen.training import sft


def homogeneous_groups(groups):
    result = []
    for group in groups:
        for sharded in (False, True):
            parameters = [p for p in group["params"] if isinstance(p, DTensor) == sharded]
            if parameters:
                result.append(dict(group, params=parameters))
    return result


def configure_optimizer(model):
    config = json.loads(
        (Path(sft.__file__).resolve().parents[1] / "configs/train.json").read_text()
    )
    groups = homogeneous_groups(
        [
            dict(name=name, params=parameters, lr=config["learning_rates"][name])
            for name, parameters in model.optimizer_parameter_groups().items()
        ]
    )
    optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    ratios = [
        config["minimum_learning_rates"].get(g["name"], g["lr"] * 0.1) / g["lr"] for g in groups
    ]
    return optimizer, scheduler(optimizer, minimum_ratios=ratios)
