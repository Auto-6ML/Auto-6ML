"""Sanity run for NNCBO Jittor teacher-weight matrix."""

from __future__ import annotations

import argparse

import jittor as jt
from jittor import nn

from teachers_weight_matrix_jittor import WeightMatrix


def main() -> None:
    parser = argparse.ArgumentParser(description="NNCBO Jittor teacher-weight matrix sanity check")
    parser.add_argument("--steps", type=int, default=5, help="optimization steps")
    parser.add_argument("--batch-size", type=int, default=64, help="batch size")
    parser.add_argument("--num-classes", type=int, default=10, help="number of classes")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    args = parser.parse_args()

    jt.flags.use_cuda = 1 if jt.has_cuda else 0
    jt.set_seed(123)

    model = WeightMatrix(num_class=args.num_classes, num_teachers=3)
    optimizer = nn.Adam(model.parameters(), lr=args.lr)

    for step in range(1, args.steps + 1):
        pre_sugrl = jt.randn((args.batch_size, args.num_classes))
        pre_gca = jt.randn((args.batch_size, args.num_classes))
        pre_dgi = jt.randn((args.batch_size, args.num_classes))
        targets = jt.randint(0, args.num_classes, shape=(args.batch_size,)).int32()

        teacher_prob = model(pre_sugrl, pre_gca, pre_dgi)
        batch_indices = jt.arange(args.batch_size).int32()
        target_prob = teacher_prob[batch_indices, targets]
        safe_prob = jt.maximum(target_prob, jt.ones_like(target_prob) * 1e-6)
        loss = -jt.log(safe_prob).mean()

        optimizer.step(loss)
        print(f"step={step}, loss={float(loss.data[0]):.6f}")

    print("NNCBO Jittor teacher-weight matrix sanity check finished.")


if __name__ == "__main__":
    main()
