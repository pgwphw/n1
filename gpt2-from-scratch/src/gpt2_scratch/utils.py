from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

import torch


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def cosine_learning_rate(
    step: int,
    max_steps: int,
    warmup_steps: int,
    learning_rate: float,
    min_learning_rate: float,
) -> float:
    if step < warmup_steps:
        return learning_rate * (step + 1) / max(1, warmup_steps)
    if step >= max_steps:
        return min_learning_rate

    decay_ratio = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    coefficient = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_learning_rate + coefficient * (learning_rate - min_learning_rate)


def build_optimizer(
    model: torch.nn.Module,
    learning_rate: float,
    weight_decay: float,
    beta1: float,
    beta2: float,
) -> torch.optim.AdamW:
    """Apply weight decay to matrix-like parameters, not biases/norm scales."""

    parameters = {name: param for name, param in model.named_parameters() if param.requires_grad}
    decay = [param for param in parameters.values() if param.dim() >= 2]
    no_decay = [param for param in parameters.values() if param.dim() < 2]

    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(
        groups,
        lr=learning_rate,
        betas=(beta1, beta2),
    )
