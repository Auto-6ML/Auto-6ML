from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.metrics.pairwise import cosine_similarity


def row_normalize_sparse(matrix: sp.spmatrix) -> sp.spmatrix:
    rowsum = np.asarray(matrix.sum(1)).flatten()
    inv = np.zeros_like(rowsum, dtype=np.float32)
    nonzero = rowsum != 0
    inv[nonzero] = np.power(rowsum[nonzero], -1)
    return sp.diags(inv).dot(matrix)


def normalize_sparse_adjacency(matrix: sp.spmatrix) -> sp.coo_matrix:
    matrix = sp.coo_matrix(matrix)
    rowsum = np.asarray(matrix.sum(1)).flatten()
    inv_sqrt = np.zeros_like(rowsum, dtype=np.float32)
    nonzero = rowsum != 0
    inv_sqrt[nonzero] = np.power(rowsum[nonzero], -0.5)
    degree = sp.diags(inv_sqrt)
    return matrix.dot(degree).transpose().dot(degree).tocoo()


def sparse_to_torch(matrix: sp.spmatrix) -> torch.Tensor:
    matrix = matrix.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((matrix.row, matrix.col)).astype(np.int64))
    values = torch.from_numpy(matrix.data)
    return torch.sparse_coo_tensor(indices, values, torch.Size(matrix.shape)).coalesce()


def normalize_dense_adjacency(adj: torch.Tensor, clamp_negative: bool = True) -> torch.Tensor:
    if clamp_negative:
        adj = adj.clamp_min(0.0)
    rowsum = adj.sum(dim=1)
    inv_sqrt = torch.where(rowsum > 0, torch.rsqrt(rowsum), torch.zeros_like(rowsum))
    degree = torch.diag(inv_sqrt)
    return adj.mm(degree).t().mm(degree)


def to_dense_float(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    if getattr(tensor, "is_sparse", False):
        tensor = tensor.to_dense()
    return tensor.float().to(device)


def build_feature_knn_graph(features: torch.Tensor, k: int) -> torch.Tensor:
    values = features.detach().cpu().numpy()
    similarity = cosine_similarity(values)
    num_nodes = similarity.shape[0]
    edges = []
    for node_id in range(num_nodes):
        neighbors = np.argpartition(similarity[node_id], -(k + 1))[-(k + 1):]
        for neighbor_id in neighbors:
            if neighbor_id > node_id:
                edges.append((node_id, neighbor_id))

    edge_array = np.asarray(edges, dtype=np.int32)
    if edge_array.size == 0:
        graph = sp.eye(num_nodes, dtype=np.float32)
    else:
        graph = sp.coo_matrix(
            (np.ones(edge_array.shape[0]), (edge_array[:, 0], edge_array[:, 1])),
            shape=(num_nodes, num_nodes),
            dtype=np.float32,
        )
        graph = graph + graph.T.multiply(graph.T > graph) - graph.multiply(graph.T > graph)
        graph = graph + sp.eye(graph.shape[0], dtype=np.float32)
    return sparse_to_torch(row_normalize_sparse(graph)).to_dense()
