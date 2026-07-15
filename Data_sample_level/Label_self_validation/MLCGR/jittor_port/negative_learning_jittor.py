"""Jittor implementation of the MLCGR negative-learning loss."""

from __future__ import annotations

import jittor as jt
from jittor import nn


def negative_learning_loss(logits: jt.Var, forbidden_labels: jt.Var, eps: float = 1e-6) -> jt.Var:
    probs = nn.softmax(logits, dim=1)
    forbidden_labels = forbidden_labels.int32()
    batch_indices = jt.arange(logits.shape[0]).int32()
    forbidden_prob = probs[batch_indices, forbidden_labels]
    safe_prob = jt.maximum(1.0 - forbidden_prob, jt.ones_like(forbidden_prob) * eps)
    return -jt.log(safe_prob).mean()
