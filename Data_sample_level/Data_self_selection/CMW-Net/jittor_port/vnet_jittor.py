"""Jittor implementation of the CMW-Net meta-weight VNet module."""

from __future__ import annotations

import jittor as jt
from jittor import nn


class VNet(nn.Module):
    """Two-layer meta-weight network used by CMW-Net."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def execute(self, x: jt.Var) -> jt.Var:
        x = nn.relu(self.linear1(x))
        return jt.sigmoid(self.linear2(x))

