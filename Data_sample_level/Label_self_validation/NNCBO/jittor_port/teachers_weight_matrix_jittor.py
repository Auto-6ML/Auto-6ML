"""Jittor implementation of NNCBO teacher weight matrix."""

from __future__ import annotations

import jittor as jt
from jittor import nn


class WeightMatrix(nn.Module):
    def __init__(self, num_class: int, num_teachers: int = 3):
        super().__init__()
        self.weight = nn.Parameter(jt.ones((num_teachers, num_class)))

    def execute(self, pre_sugrl: jt.Var, pre_gca: jt.Var, pre_dgi: jt.Var) -> jt.Var:
        multi_teacher_pre = jt.stack((pre_sugrl, pre_gca, pre_dgi), dim=1).stop_grad()
        teacher_pre = (multi_teacher_pre * self.weight.unsqueeze(0)).sum(dim=1)
        return nn.softmax(teacher_pre, dim=1)
