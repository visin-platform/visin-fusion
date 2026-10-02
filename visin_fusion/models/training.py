"""Shared optimizer and schedule building blocks for model-owned training defaults."""

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class TrainingSetup:
    """What the training engine needs from a model.

    An optimizer, an optional schedule, a loss ``(raw_outputs, logits, labels) -> scalar``, an optional
    gradient-clipping norm and whether to use mixed precision.
    """

    optimizer: torch.optim.Optimizer
    scheduler: object | None
    loss: Callable
    clip_grad_norm: float | None = None
    mixed_precision: bool = True


def weighted_cross_entropy(class_weights, device):
    """A loss callable that applies class-weighted cross-entropy to the dense logits."""
    ce = nn.CrossEntropyLoss(weight=torch.as_tensor(class_weights, dtype=torch.float32)).to(device)
    return lambda outputs, segmap, labels: ce(segmap, labels)


def warmup_then(optimizer, warmup_epochs, start_factor, main):
    """A linear warmup from ``start_factor`` for ``warmup_epochs``, then the ``main`` schedule."""
    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=start_factor, total_iters=warmup_epochs)
    return torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup, main], [warmup_epochs])


def query_schedule(optimizer, options, epochs):
    """The polynomial (or step) schedule after warmup used by the query models."""
    warmup = options.get("warmup_epochs", 10)
    post_warmup = max(1, epochs - warmup)
    if options.get("lr_scheduler", "poly") == "step":
        milestones = [max(1, int(f * post_warmup)) for f in options.get("lr_step_milestones", [0.75, 0.90])]
        main = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=options.get("lr_step_gamma", 0.1)
        )
    else:
        main = torch.optim.lr_scheduler.PolynomialLR(
            optimizer, total_iters=post_warmup, power=options.get("lr_poly_power", 0.9)
        )
    return warmup_then(optimizer, warmup, 1e-6, main)


def backbone_slower(model, lr):
    """AdamW with the backbone at a tenth of ``lr``, the rest at ``lr``."""
    backbone = list(model.backbone.parameters())
    ids = {id(p) for p in backbone}
    rest = [p for p in model.parameters() if id(p) not in ids]
    return torch.optim.AdamW([{"params": backbone, "lr": lr * 0.1}, {"params": rest, "lr": lr}], weight_decay=0.05)


def deeplab_schedule(optimizer, options, epochs):
    """The scheduler named by ``options['lr_scheduler']`` (step, cosine, exponential or plateau), or ``None``."""
    spec = options.get("lr_scheduler")
    if not spec:
        return None
    kind = spec.get("type", "step")
    schedulers = torch.optim.lr_scheduler
    if kind == "step":
        return schedulers.StepLR(optimizer, step_size=spec.get("step_size", 30), gamma=spec.get("gamma", 0.1))
    if kind == "cosine":
        return schedulers.CosineAnnealingLR(
            optimizer, T_max=spec.get("T_max", epochs), eta_min=spec.get("eta_min", 1e-6)
        )
    if kind == "exponential":
        return schedulers.ExponentialLR(optimizer, gamma=spec.get("gamma", 0.95))
    if kind == "plateau":
        return schedulers.ReduceLROnPlateau(
            optimizer,
            mode=spec.get("mode", "max"),
            factor=spec.get("factor", 0.1),
            patience=spec.get("patience", 10),
            min_lr=spec.get("min_lr", 1e-6),
        )
    raise ValueError(f"DeepLabV3Plus.lr_scheduler.type {kind!r} is not one of step, cosine, exponential, plateau")
