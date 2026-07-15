"""Sanity run for the Jittor VNet port."""

from __future__ import annotations

import argparse

import jittor as jt
from jittor import nn

from vnet_jittor import VNet


def main() -> None:
    parser = argparse.ArgumentParser(description="CMW-Net Jittor VNet sanity check")
    parser.add_argument("--steps", type=int, default=5, help="optimization steps")
    parser.add_argument("--batch-size", type=int, default=64, help="batch size")
    parser.add_argument("--input-dim", type=int, default=1, help="VNet input dimension")
    parser.add_argument("--hidden-dim", type=int, default=100, help="VNet hidden dimension")
    parser.add_argument("--output-dim", type=int, default=1, help="VNet output dimension")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    args = parser.parse_args()

    jt.flags.use_cuda = 1 if jt.has_cuda else 0
    jt.set_seed(123)

    model = VNet(args.input_dim, args.hidden_dim, args.output_dim)
    optimizer = nn.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    for step in range(1, args.steps + 1):
        inputs = jt.randn((args.batch_size, args.input_dim))
        targets = jt.sigmoid(inputs)
        preds = model(inputs)
        loss = criterion(preds, targets)
        optimizer.step(loss)
        print(f"step={step}, loss={float(loss.data[0]):.6f}")

    print("Jittor VNet sanity check finished.")


if __name__ == "__main__":
    main()

