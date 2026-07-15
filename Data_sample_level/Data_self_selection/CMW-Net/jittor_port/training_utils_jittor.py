"""Phase-2/3 Jittor training helpers for CMW-Net migration."""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import numpy as np
import jittor as jt
from jittor import nn


class KeepWeightLoss:
    """Track per-sample losses and learned weights."""

    def __init__(self, num_samples: int):
        self.loss = jt.zeros((num_samples, 1), dtype=jt.float32)
        self.weight = jt.zeros((num_samples,), dtype=jt.float32)

    def __call__(self, epoch_loss: jt.Var, epoch_weight: jt.Var, index: jt.Var) -> None:
        self.loss[index] = epoch_loss.stop_grad()
        self.weight[index] = epoch_weight.stop_grad()


class SelfAdaptiveTrainingCE:
    """Pseudo-label memory used by CMW-Net training."""

    def __init__(self, labels: Sequence[int], num_classes: int):
        labels_np = np.asarray(labels, dtype=np.int64)
        one_hot = np.zeros((labels_np.shape[0], num_classes), dtype=np.float32)
        one_hot[np.arange(labels_np.shape[0]), labels_np] = 1.0
        self.soft_labels = jt.array(one_hot)

    def __call__(self, logits: jt.Var, index: jt.Var, epoch: int) -> jt.Var:  # noqa: ARG002
        probs = nn.softmax(logits, dim=1).stop_grad()
        self.soft_labels[index] = probs
        return self.soft_labels[index]


def update_ema(model: nn.Module, model_ema: nn.Module, momentum: float) -> None:
    """EMA update used by the original training loop."""

    for param, param_ema in zip(model.parameters(), model_ema.parameters()):
        param_ema.assign(param_ema * momentum + param * (1.0 - momentum))


def train_ce_epoch(
    train_loader: Iterable[Tuple[jt.Var, jt.Var, jt.Var]],
    model: nn.Module,
    model_ema: nn.Module,
    optimizer: nn.Optimizer,
    ema_momentum: float,
) -> dict:
    """Jittor version of the `train_CE` warmup stage from cmwn.py."""

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total = 0
    correct = 0
    num_batches = 0

    for inputs, targets, _ in train_loader:
        num_batches += 1
        logits = model(inputs)
        loss = criterion(logits, targets)
        optimizer.step(loss)
        update_ema(model, model_ema, ema_momentum)

        pred = jt.argmax(logits, dim=1)
        correct += int((pred == targets).sum().item())
        total += int(targets.shape[0])
        total_loss += float(loss.item())

    avg_loss = total_loss / max(1, num_batches)
    acc = 100.0 * correct / max(1, total)
    return {"avg_loss": avg_loss, "acc": acc, "total": total}


def _cross_entropy_per_sample(logits: jt.Var, targets: jt.Var) -> jt.Var:
    """Per-sample cross-entropy used by VNet weighting."""

    log_probs = nn.log_softmax(logits, dim=1)
    row_index = jt.arange(targets.shape[0]).int32()
    return -log_probs[row_index, targets]


def weighted_ce_loss_with_vnet(
    logits: jt.Var,
    targets: jt.Var,
    vnet: nn.Module,
    detach_weight: bool = False,
) -> Tuple[jt.Var, jt.Var]:
    """Apply VNet weights on per-sample CE losses."""

    per_sample_loss = _cross_entropy_per_sample(logits, targets).reshape((-1, 1))
    weight = vnet(per_sample_loss.stop_grad())
    if detach_weight:
        weight = weight.stop_grad()
    loss = (per_sample_loss * weight).mean()
    return loss, weight


def train_meta_epoch(
    train_loader: Iterable[Tuple[jt.Var, jt.Var, jt.Var]],
    meta_loader: Iterable[Tuple[jt.Var, jt.Var]],
    model: nn.Module,
    model_ema: nn.Module,
    vnet: nn.Module,
    optimizer_model: nn.Optimizer,
    optimizer_vnet: nn.Optimizer,
    ema_momentum: float,
) -> dict:
    """Jittor phase-3 style meta-training loop with VNet sample weighting."""

    total_loss = 0.0
    total_meta_loss = 0.0
    total_weight = 0.0
    total = 0
    correct = 0
    num_batches = 0
    num_meta_batches = 0
    meta_iter = iter(meta_loader)

    for inputs, targets, _ in train_loader:
        num_batches += 1

        logits = model(inputs)
        train_loss, train_weight = weighted_ce_loss_with_vnet(
            logits, targets, vnet, detach_weight=True
        )
        optimizer_model.step(train_loss)
        update_ema(model, model_ema, ema_momentum)

        pred = jt.argmax(logits, dim=1)
        correct += int((pred == targets).sum().item())
        total += int(targets.shape[0])
        total_loss += float(train_loss.item())
        total_weight += float(train_weight.mean().item())

        try:
            meta_inputs, meta_targets = next(meta_iter)
        except StopIteration:
            meta_iter = iter(meta_loader)
            meta_inputs, meta_targets = next(meta_iter)

        meta_logits = model(meta_inputs).stop_grad()
        meta_loss, _ = weighted_ce_loss_with_vnet(meta_logits, meta_targets, vnet)
        optimizer_vnet.step(meta_loss)
        total_meta_loss += float(meta_loss.item())
        num_meta_batches += 1

    avg_loss = total_loss / max(1, num_batches)
    avg_meta_loss = total_meta_loss / max(1, num_meta_batches)
    acc = 100.0 * correct / max(1, total)
    avg_weight = total_weight / max(1, num_batches)
    return {
        "avg_loss": avg_loss,
        "avg_meta_loss": avg_meta_loss,
        "avg_weight": avg_weight,
        "acc": acc,
        "total": total,
    }
