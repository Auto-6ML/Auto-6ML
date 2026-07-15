"""Phase-2 demo: Jittor `train_CE` style loop for CMW-Net."""

from __future__ import annotations

import argparse
import copy
from typing import Iterator, Tuple

import numpy as np
import jittor as jt
from jittor import nn

from training_utils_jittor import KeepWeightLoss, SelfAdaptiveTrainingCE, train_ce_epoch


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


def main() -> None:
    parser = argparse.ArgumentParser(description="CMW-Net phase-2 Jittor train_CE demo")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--ema", type=float, default=0.997)
    args = parser.parse_args()

    jt.flags.use_cuda = 1 if jt.has_cuda else 0
    jt.set_seed(123)

    train_loader = SyntheticIndexedLoader(
        num_samples=args.samples,
        batch_size=args.batch_size,
        num_classes=args.num_classes,
    )
    model = TinyClassifier(args.num_classes)
    model_ema = copy.deepcopy(model)
    optimizer = nn.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    sat = SelfAdaptiveTrainingCE(labels=train_loader.labels, num_classes=args.num_classes)
    wl_tracker = KeepWeightLoss(num_samples=args.samples)
    for epoch in range(1, args.epochs + 1):
        metrics = train_ce_epoch(train_loader, model, model_ema, optimizer, args.ema)

        for inputs, targets, index in train_loader:
            logits_ema = model_ema(inputs)
            _ = sat(logits_ema, index, epoch=epoch)
            logits = model(inputs)
            per_sample_loss = jt.mean(jt.abs(logits), dim=1).reshape((-1, 1))
            pseudo_conf = jt.ones((index.shape[0],), dtype=jt.float32)
            wl_tracker(per_sample_loss, pseudo_conf, index)

        tracked_mean_loss = float(wl_tracker.loss.mean().item())
        print(
            f"epoch={epoch}, avg_loss={metrics['avg_loss']:.6f}, "
            f"acc={metrics['acc']:.2f}, tracked_loss={tracked_mean_loss:.6f}"
        )

    print("Phase-2 Jittor train_CE demo finished.")


if __name__ == "__main__":
    main()
