from __future__ import annotations

import argparse
from pathlib import Path

from mgbo.trainer import TrainOptions, train


DATASETS = ["acm", "dblp", "imdb", "freebase"]


def parse_args() -> TrainOptions:
    project_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description="Run MGBO with differentiable bilevel optimization.")
    parser.add_argument("-d", "--dataset", choices=DATASETS, default="acm")
    parser.add_argument("--config-dir", type=Path, default=Path("configs"))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--use-pretrain",
        "--use_pretrain",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Evaluate a pretrained MGBO checkpoint after training.",
    )
    parser.add_argument("--eval-only", action="store_true", help="Skip training and evaluate a saved checkpoint.")
    parser.add_argument(
        "--pretrain-dir",
        type=Path,
        default=None,
        help="Directory containing {dataset}.model.pt and {dataset}.graph.pt.",
    )
    parser.add_argument(
        "--pretrain-epoch",
        type=int,
        default=None,
        help="Checkpoint epoch to load from --pretrain-dir or --output-dir.",
    )
    parser.add_argument(
        "--pretrain-model-path",
        type=Path,
        default=None,
        help="Explicit pretrained model state_dict path.",
    )
    parser.add_argument(
        "--pretrain-graph-path",
        type=Path,
        default=None,
        help="Explicit pretrained graph tensor path.",
    )
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    args = parser.parse_args()

    return TrainOptions(
        dataset=args.dataset,
        project_dir=project_dir,
        config_dir=Path(args.config_dir),
        output_dir=Path(args.output_dir),
        seed=args.seed,
        no_cuda=args.no_cuda,
        use_pretrain=args.use_pretrain,
        eval_only=args.eval_only,
        pretrain_dir=args.pretrain_dir,
        pretrain_epoch=args.pretrain_epoch,
        pretrain_model_path=args.pretrain_model_path,
        pretrain_graph_path=args.pretrain_graph_path,
    )


if __name__ == "__main__":
    train(parse_args())
