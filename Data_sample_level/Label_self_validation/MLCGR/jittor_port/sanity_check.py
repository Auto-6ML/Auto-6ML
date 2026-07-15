"""Sanity run for MLCGR Jittor MetaNet and negative-learning loss."""

from __future__ import annotations

import argparse

import jittor as jt
from jittor import nn

from meta_net_jittor import MetaNet
from negative_learning_jittor import negative_learning_loss


def main() -> None:
    parser = argparse.ArgumentParser(description="MLCGR Jittor sanity check")
    parser.add_argument("--steps", type=int, default=5, help="optimization steps")
    parser.add_argument("--batch-size", type=int, default=64, help="batch size")
    parser.add_argument("--feature-dim", type=int, default=64, help="input feature dimension")
    parser.add_argument("--hidden-dim", type=int, default=200, help="hidden dimension")
    parser.add_argument("--num-classes", type=int, default=10, help="number of classes")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    args = parser.parse_args()

    jt.flags.use_cuda = 1 if jt.has_cuda else 0
    jt.set_seed(123)

    model = MetaNet(args.feature_dim, args.hidden_dim, args.num_classes)
    optimizer = nn.Adam(model.parameters(), lr=args.lr)

    for step in range(1, args.steps + 1):
        features = jt.randn((args.batch_size, args.feature_dim))
        labels = jt.randint(0, args.num_classes, shape=(args.batch_size,)).int32()
        forbidden_labels = ((labels + 1) % args.num_classes).int32()

        logits = model(features)
        ce_loss = nn.cross_entropy_loss(logits, labels)
        nl_loss = negative_learning_loss(logits, forbidden_labels)
        loss = ce_loss + nl_loss

        optimizer.step(loss)
        print(
            f"step={step}, ce_loss={float(ce_loss.data[0]):.6f}, "
            f"nl_loss={float(nl_loss.data[0]):.6f}, total={float(loss.data[0]):.6f}"
        )

    print("MLCGR Jittor sanity check finished.")


if __name__ == "__main__":
    main()
