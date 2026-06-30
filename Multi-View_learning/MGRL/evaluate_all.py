from __future__ import annotations

import argparse
import json
from pathlib import Path

from mgbo.trainer import TrainOptions, train


DATASETS = ("acm", "dblp", "imdb", "freebase")

PAPER_TARGETS = {
    "acm": {"macro_f1": 0.933, "micro_f1": 0.932},
    "dblp": {"macro_f1": 0.952, "micro_f1": 0.956},
    "imdb": {"macro_f1": 0.607, "micro_f1": 0.614},
    "freebase": {"macro_f1": 0.615, "micro_f1": 0.675},
}

def _resolve(project_dir: Path, path: Path) -> Path:
    return path if path.is_absolute() else project_dir / path


def _load_options(
    project_dir: Path,
    config_dir: Path,
    dataset: str,
    pretrain_dir: Path,
    no_cuda: bool,
) -> TrainOptions:
    return TrainOptions(
        dataset=dataset,
        project_dir=project_dir,
        config_dir=config_dir,
        use_pretrain=True,
        eval_only=True,
        pretrain_dir=pretrain_dir,
        no_cuda=no_cuda,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate released MGBO checkpoints.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DATASETS,
        default=list(DATASETS),
        help="Datasets to evaluate.",
    )
    parser.add_argument("--config-dir", type=Path, default=Path("configs"))
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_dir = Path(__file__).resolve().parent
    pretrain_dir = Path("pretrained")

    results: dict[str, dict[str, float]] = {}
    for dataset in args.datasets:
        options = _load_options(project_dir, args.config_dir, dataset, pretrain_dir, args.no_cuda)
        print(f"\n=== {dataset.upper()} | pretrain_dir={pretrain_dir} ===")
        results[dataset] = train(options)

    print("\nSummary")
    print("dataset    macro_f1  micro_f1  paper_macro  paper_micro  macro_delta  micro_delta")
    for dataset in args.datasets:
        metrics = results[dataset]
        target = PAPER_TARGETS[dataset]
        macro_delta = metrics["macro_f1"] - target["macro_f1"]
        micro_delta = metrics["micro_f1"] - target["micro_f1"]
        print(
            f"{dataset:<10} "
            f"{metrics['macro_f1']:.4f}    "
            f"{metrics['micro_f1']:.4f}    "
            f"{target['macro_f1']:.4f}       "
            f"{target['micro_f1']:.4f}       "
            f"{macro_delta:+.4f}       "
            f"{micro_delta:+.4f}"
        )

    if args.output_json is not None:
        output_path = _resolve(project_dir, args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        serialisable = {
            dataset: {
                **metrics,
                "paper_macro_f1": PAPER_TARGETS[dataset]["macro_f1"],
                "paper_micro_f1": PAPER_TARGETS[dataset]["micro_f1"],
            }
            for dataset, metrics in results.items()
        }
        output_path.write_text(json.dumps(serialisable, indent=2), encoding="utf-8")
        print(f"\nSaved JSON summary to {output_path}")


if __name__ == "__main__":
    main()
