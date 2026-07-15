"""Jittor implementation of MLCGR MetaNet."""

from __future__ import annotations

import jittor as jt
from jittor import nn


class MetaNet(nn.Module):
    def __init__(self, feature_dim: int = 64, hidden_dim: int = 200, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def execute(self, features: jt.Var) -> jt.Var:
        return self.net(features)
