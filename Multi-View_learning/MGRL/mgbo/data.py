from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F

from .graph_ops import normalize_sparse_adjacency, sparse_to_torch


@dataclass
class DatasetBundle:
    name: str
    metapaths: list[torch.Tensor]
    features: torch.Tensor
    labels: torch.Tensor
    num_classes: int
    train_idx: torch.Tensor
    val_idx: torch.Tensor
    test_idx: torch.Tensor


def _indices(split_file, key: str) -> torch.Tensor:
    return torch.from_numpy(split_file[key]).long()


def load_acm(data_dir: str | Path) -> DatasetBundle:
    prefix = Path(data_dir) / "ACM"
    features = sp.load_npz(prefix / "features_0.npz").toarray()
    labels = np.load(prefix / "labels.npy")
    split = np.load(prefix / "train_val_test_idx.npz")

    scale = 2
    pap = sp.load_npz(prefix / "pap.npz").toarray()
    pap = torch.as_tensor(pap + np.eye(pap.shape[0]) * scale, dtype=torch.float32)
    pap = F.normalize(pap, dim=1, p=2)

    psp = sp.load_npz(prefix / "psp.npz").toarray()
    psp = torch.as_tensor(psp + np.eye(psp.shape[0]) * scale, dtype=torch.float32)
    psp = F.normalize(psp, dim=1, p=2)

    return DatasetBundle(
        name="acm",
        metapaths=[pap, psp],
        features=torch.as_tensor(features, dtype=torch.float32),
        labels=torch.as_tensor(labels, dtype=torch.long),
        num_classes=3,
        train_idx=_indices(split, "train_idx"),
        val_idx=_indices(split, "val_idx"),
        test_idx=_indices(split, "test_idx"),
    )


def load_dblp(data_dir: str | Path) -> DatasetBundle:
    prefix = Path(data_dir) / "DBLP"
    features = sp.load_npz(prefix / "features_0.npz").toarray()
    labels = np.load(prefix / "labels.npy")
    split = np.load(prefix / "train_val_test_idx.npz")

    metapaths = []
    for filename in ["apa.npz", "aptpa.npz", "apvpa.npz"]:
        adj = sp.load_npz(prefix / filename).toarray()
        adj = torch.as_tensor(adj, dtype=torch.float32)
        metapaths.append(F.normalize(adj, dim=1, p=2))

    return DatasetBundle(
        name="dblp",
        metapaths=metapaths,
        features=torch.as_tensor(features, dtype=torch.float32),
        labels=torch.as_tensor(labels, dtype=torch.long),
        num_classes=4,
        train_idx=_indices(split, "train_idx"),
        val_idx=_indices(split, "val_idx"),
        test_idx=_indices(split, "test_idx"),
    )


def load_imdb(data_dir: str | Path) -> DatasetBundle:
    path = Path(data_dir) / "imdb.pkl"
    with path.open("rb") as handle:
        data = pickle.load(handle)

    scale = 3
    mdm = data["MDM"] + np.eye(data["MDM"].shape[0]) * scale
    mam = data["MAM"] + np.eye(data["MAM"].shape[0]) * scale

    return DatasetBundle(
        name="imdb",
        metapaths=[
            sparse_to_torch(normalize_sparse_adjacency(mdm)),
            sparse_to_torch(normalize_sparse_adjacency(mam)),
        ],
        features=torch.as_tensor(data["feature"].astype(float), dtype=torch.float32),
        labels=torch.argmax(torch.as_tensor(data["label"], dtype=torch.float32), dim=1).long(),
        num_classes=3,
        train_idx=torch.from_numpy(data["train_idx"].ravel()).long(),
        val_idx=torch.from_numpy(data["val_idx"].ravel()).long(),
        test_idx=torch.from_numpy(data["test_idx"].ravel()).long(),
    )


def _preprocess_features(features: sp.spmatrix) -> np.ndarray:
    rowsum = np.asarray(features.sum(1)).flatten()
    inv = np.zeros_like(rowsum, dtype=np.float32)
    nonzero = rowsum != 0
    inv[nonzero] = np.power(rowsum[nonzero], -1)
    return sp.diags(inv).dot(features).todense()


def load_freebase(data_dir: str | Path) -> DatasetBundle:
    prefix = Path(data_dir) / "freebase"
    labels = np.load(prefix / "labels.npy").astype(np.int64)
    features = _preprocess_features(sp.eye(3492, dtype=np.float32))

    return DatasetBundle(
        name="freebase",
        metapaths=[
            sparse_to_torch(normalize_sparse_adjacency(sp.load_npz(prefix / "mdm.npz"))),
            sparse_to_torch(normalize_sparse_adjacency(sp.load_npz(prefix / "mam.npz"))),
            sparse_to_torch(normalize_sparse_adjacency(sp.load_npz(prefix / "mwm.npz"))),
        ],
        features=torch.as_tensor(features, dtype=torch.float32),
        labels=torch.as_tensor(labels, dtype=torch.long),
        num_classes=4,
        train_idx=torch.from_numpy(np.load(prefix / "train_60.npy").astype(np.int64)).long(),
        val_idx=torch.from_numpy(np.load(prefix / "val_60.npy").astype(np.int64)).long(),
        test_idx=torch.from_numpy(np.load(prefix / "test_60.npy").astype(np.int64)).long(),
    )


LOADERS = {
    "acm": load_acm,
    "dblp": load_dblp,
    "imdb": load_imdb,
    "freebase": load_freebase,
}


def load_dataset(name: str, data_dir: str | Path) -> DatasetBundle:
    try:
        loader = LOADERS[name]
    except KeyError as exc:
        choices = ", ".join(sorted(LOADERS))
        raise ValueError(f"Unknown dataset '{name}'. Choices: {choices}") from exc
    return loader(data_dir)
