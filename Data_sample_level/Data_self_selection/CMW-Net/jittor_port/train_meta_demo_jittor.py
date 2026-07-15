"""Phase-3 demo: Jittor meta-training loop with VNet weighting for CMW-Net."""

from __future__ import annotations

import argparse
import copy
from typing import Iterator, Tuple

import numpy as np
import jittor as jt
from jittor import nn

from training_utils_jittor import train_ce_epoch, train_meta_epoch
from vnet_jittor import VNet


class TinyClassifier(nn.Module):
    """Small classifier used for migration sanity checks."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 32 * 32, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def execute(self, x: jt.Var) -> jt.Var:
        return self.net(x)


class SyntheticIndexedLoader:
    """Produces `(inputs, targets, index)` like original CMW-Net loader."""

    def __init__(self, num_samples: int, batch_size: int, num_classes: int):
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.labels = np.random.randint(0, num_classes, size=(num_samples,), dtype=np.int64)

    def __iter__(self) -> Iterator[Tuple[jt.Var, jt.Var, jt.Var]]:
        order = np.random.permutation(self.num_samples)
        for start in range(0, self.num_samples, self.batch_size):
            idx = order[start : start + self.batch_size]
            inputs = jt.randn((idx.shape[0], 3, 32, 32))
            targets = jt.array(self.labels[idx]).int32()
            index = jt.array(idx).int32()
            yield inputs, targets, index


class SyntheticMetaLoader:
    """Produces `(inputs, targets)` meta batches for VNet updates."""

    def __init__(self, num_samples: int, batch_size: int, num_classes: int):
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.labels = np.random.randint(0, num_classes, size=(num_samples,), dtype=np.int64)

    def __iter__(self) -> Iterator[Tuple[jt.Var, jt.Var]]:
        order = np.random.permutation(self.num_samples)
        for start in range(0, self.num_samples, self.batch_size):
            idx = order[start : start + self.batch_size]
            inputs = jt.randn((idx.shape[0], 3, 32, 32))
            targets = jt.array(self.labels[idx]).int32()
            yield inputs, targets


def main() -> None:
    parser = argparse.ArgumentParser(description="CMW-Net phase-3 Jittor meta-training demo")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--meta-samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--vnet-lr", type=float, default=1e-3)
    parser.add_argument("--ema", type=float, default=0.997)
    args = parser.parse_args()

    jt.flags.use_cuda = 1 if jt.has_cuda else 0
    jt.set_seed(123)

    train_loader = SyntheticIndexedLoader(
        num_samples=args.samples,
        batch_size=args.batch_size,
        num_classes=args.num_classes,
    )
    meta_loader = SyntheticMetaLoader(
        num_samples=args.meta_samples,
        batch_size=args.batch_size,
        num_classes=args.num_classes,
    )

    model = TinyClassifier(args.num_classes)
    model_ema = copy.deepcopy(model)
    vnet = VNet(input_dim=1, hidden_dim=100, output_dim=1)

    optimizer_model = nn.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    optimizer_vnet = nn.Adam(vnet.parameters(), lr=args.vnet_lr, weight_decay=1e-4)

    for epoch in range(1, args.epochs + 1):
        if epoch <= args.warmup_epochs:
            metrics = train_ce_epoch(train_loader, model, model_ema, optimizer_model, args.ema)
            print(
                f"epoch={epoch}, stage=warmup, avg_loss={metrics['avg_loss']:.6f}, "
                f"acc={metrics['acc']:.2f}"
            )
            continue

        metrics = train_meta_epoch(
            train_loader=train_loader,
            meta_loader=meta_loader,
            model=model,
            model_ema=model_ema,
            vnet=vnet,
            optimizer_model=optimizer_model,
            optimizer_vnet=optimizer_vnet,
            ema_momentum=args.ema,
        )
        print(
            f"epoch={epoch}, stage=meta, avg_loss={metrics['avg_loss']:.6f}, "
            f"avg_meta_loss={metrics['avg_meta_loss']:.6f}, "
            f"avg_weight={metrics['avg_weight']:.6f}, acc={metrics['acc']:.2f}"
        )

    print("Phase-3 Jittor meta-training demo finished.")


if __name__ == "__main__":
    main()
