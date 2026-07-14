from __future__ import annotations

import torch
import torch.nn.functional as F

from .graph_ops import normalize_dense_adjacency


def graph_structure_loss(
    C: torch.Tensor,
    embeddings: torch.Tensor,
    metapaths: list[torch.Tensor],
    reconstruction_weight: float,
    regularization_weight: float,
    metapath_weights: tuple[float, ...],
) -> torch.Tensor:
    reconstruction = torch.linalg.matrix_norm(embeddings.t() - embeddings.t().matmul(C), ord="fro")
    regularization = torch.linalg.vector_norm(C, ord=2)
    differences = [torch.linalg.vector_norm(C - metapath, ord=2) for metapath in metapaths]

    weights = list(metapath_weights)
    if len(weights) != len(differences):
        raise ValueError(f"Expected {len(differences)} metapath weights, got {len(weights)}")

    loss = reconstruction_weight * reconstruction + regularization_weight * regularization
    for weight, difference in zip(weights, differences):
        loss = loss + weight * difference
    return loss


def build_learned_graph(
    C: torch.Tensor,
    initial_graph: torch.Tensor,
    initial_graph_weight: float,
    normalization: str,
) -> torch.Tensor:
    graph = (C + C.t()) / 2
    learned_graph_weight = 1.0 - initial_graph_weight

    if normalization == "adjacency":
        return learned_graph_weight * normalize_dense_adjacency(graph) + initial_graph_weight * initial_graph
    if normalization == "l2":
        return learned_graph_weight * F.normalize(graph, dim=1, p=2) + initial_graph_weight * initial_graph
    raise ValueError(f"Unknown learned graph normalization: {normalization}")
