from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .graph_ops import normalize_dense_adjacency


class GraphConvolution(nn.Module):
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.0):
        super().__init__()
        self.dropout = dropout
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        features = F.dropout(features, self.dropout, training=self.training)
        support = features.mm(self.weight)
        if getattr(adj, "is_sparse", False):
            output = torch.sparse.mm(adj, support)
        else:
            output = adj.mm(support)
        return F.relu(output)


class ChannelAttention(nn.Module):
    def __init__(self, num_channels: int):
        super().__init__()
        self.weight = nn.Parameter(torch.full((num_channels, 1, 1), 0.1))

    def forward(self, matrices: list[torch.Tensor], norm_p: int | float = 0) -> torch.Tensor:
        dense_matrices = [matrix.to_dense() if getattr(matrix, "is_sparse", False) else matrix for matrix in matrices]
        stacked = torch.stack(dense_matrices)
        if norm_p:
            stacked = F.normalize(stacked, dim=1, p=norm_p)
        attention = F.softmax(self.weight, dim=0)
        return torch.sum(stacked * attention, dim=0)


class MGBOModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_classes: int,
        num_metapaths: int,
        dropout: float = 0.0,
        graph_clamp_negative: bool = True,
    ):
        super().__init__()
        self.gcn = GraphConvolution(input_size, hidden_size, dropout=dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)
        self.topology_attention = ChannelAttention(num_metapaths)
        self.structure_attention = ChannelAttention(2)
        self.graph_clamp_negative = graph_clamp_negative

    def forward(
        self,
        features: torch.Tensor,
        learned_graph: torch.Tensor,
        metapaths: list[torch.Tensor],
        topology_norm: int | float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        topology = self.topology_attention(metapaths, topology_norm)
        topology = normalize_dense_adjacency(topology.t() + topology)

        graph = self.structure_attention([learned_graph, topology], norm_p=1)
        graph = normalize_dense_adjacency(graph.t() + graph, clamp_negative=self.graph_clamp_negative)

        embeddings = self.gcn(features, graph)
        logits = self.classifier(embeddings)
        return F.log_softmax(logits, dim=1), embeddings
