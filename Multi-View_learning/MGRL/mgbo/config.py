from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class DataConfig:
    n: int
    fdim: int
    class_num: int


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int
    inner_steps: int
    inner_lr: float
    patience: int
    dropout: float
    lr: float
    weight_decay: float
    hidden_size: int
    knn_k: int
    topology_norm: int | float
    learned_graph_norm: str
    reconstruction_weight: float
    regularization_weight: float
    metapath_weights: tuple[float, ...]
    initial_graph_weight: float
    min_epochs: int
    graph_clamp_negative: bool


@dataclass(frozen=True)
class DatasetConfig:
    data: DataConfig
    training: TrainingConfig


REQUIRED_TRAINING_KEYS = {field.name for field in fields(TrainingConfig)}


def dataset_config_key(dataset: str) -> str:
    return f"MGBO-{dataset}"


def find_dataset_config(config_dir: str | Path, dataset: str) -> Path:
    config_dir = Path(config_dir)
    for filename in ("args.yaml", "args.yml"):
        path = config_dir / filename
        if path.exists():
            return path
    for suffix in (".yaml", ".yml"):
        path = config_dir / f"{dataset}{suffix}"
        if path.exists():
            return path
    raise FileNotFoundError(f"Cannot find config for dataset '{dataset}' in {config_dir}")


def load_config(path: str | Path, dataset: str | None = None) -> DataConfig:
    return load_dataset_config(path, dataset).data


def load_dataset_config(path: str | Path, dataset: str | None = None) -> DatasetConfig:
    path = Path(path)
    if path.suffix.lower() in {".yaml", ".yml"}:
        return _load_yaml_config(path, dataset)
    raise ValueError(f"Unsupported config file extension: {path.suffix}")


def _load_yaml_config(path: Path, dataset: str | None) -> DatasetConfig:
    raw = _read_yaml(path)
    if not isinstance(raw, Mapping):
        raise ValueError(f"YAML config must be a mapping: {path}")

    config = _select_yaml_entry(raw, path, dataset)
    data = _section(config, "data", path)
    training = _section(config, "training", path)
    missing = sorted(key for key in REQUIRED_TRAINING_KEYS if key not in training or training[key] is None)
    if missing:
        names = ", ".join(missing)
        raise ValueError(f"Missing required training values in {path}: {names}")

    return DatasetConfig(
        data=DataConfig(
            n=_as_int(data, "n", path),
            fdim=_as_int(data, "fdim", path),
            class_num=_as_int(data, "class_num", path),
        ),
        training=TrainingConfig(
            epochs=_as_int(training, "epochs", path),
            inner_steps=_as_int(training, "inner_steps", path),
            inner_lr=_as_float(training, "inner_lr", path),
            patience=_as_int(training, "patience", path),
            dropout=_as_float(training, "dropout", path),
            lr=_as_float(training, "lr", path),
            weight_decay=_as_float(training, "weight_decay", path),
            hidden_size=_as_int(training, "hidden_size", path),
            knn_k=_as_int(training, "knn_k", path),
            topology_norm=_as_number(training, "topology_norm", path),
            learned_graph_norm=_as_choice(training, "learned_graph_norm", {"l2", "adjacency"}, path),
            reconstruction_weight=_as_float(training, "reconstruction_weight", path),
            regularization_weight=_as_float(training, "regularization_weight", path),
            metapath_weights=_as_float_tuple(training, "metapath_weights", path),
            initial_graph_weight=_as_float(training, "initial_graph_weight", path),
            min_epochs=_as_int(training, "min_epochs", path),
            graph_clamp_negative=_as_bool(training, "graph_clamp_negative", path),
        ),
    )


def _select_yaml_entry(raw: Mapping[str, Any], path: Path, dataset: str | None) -> Mapping[str, Any]:
    if "data" in raw and "training" in raw:
        return raw

    if dataset is None:
        raise ValueError(f"Dataset must be provided when loading centralized YAML config: {path}")

    key = dataset_config_key(dataset)
    entry = raw.get(key)
    if not isinstance(entry, Mapping):
        raise KeyError(f"Cannot find YAML config key '{key}' in {path}")
    return entry


def _read_yaml(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        try:
            import yaml

            return yaml.safe_load(file)
        except ImportError:
            file.seek(0)
            from ruamel.yaml import YAML

            return YAML(typ="safe").load(file)


def _section(raw: Mapping[str, Any], name: str, path: Path) -> Mapping[str, Any]:
    value = raw.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"Missing or invalid '{name}' section in {path}")
    return value


def _required(section: Mapping[str, Any], key: str, path: Path) -> Any:
    if key not in section:
        raise ValueError(f"Missing '{key}' in {path}")
    return section[key]


def _as_int(section: Mapping[str, Any], key: str, path: Path) -> int:
    return int(_required(section, key, path))


def _as_float(section: Mapping[str, Any], key: str, path: Path) -> float:
    return float(_required(section, key, path))


def _as_number(section: Mapping[str, Any], key: str, path: Path) -> int | float:
    value = _required(section, key, path)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"'{key}' must be numeric in {path}")
    return value


def _as_bool(section: Mapping[str, Any], key: str, path: Path) -> bool:
    value = _required(section, key, path)
    if not isinstance(value, bool):
        raise ValueError(f"'{key}' must be boolean in {path}")
    return value


def _as_choice(section: Mapping[str, Any], key: str, choices: set[str], path: Path) -> str:
    value = str(_required(section, key, path))
    if value not in choices:
        valid = ", ".join(sorted(choices))
        raise ValueError(f"'{key}' must be one of: {valid}")
    return value


def _as_float_tuple(section: Mapping[str, Any], key: str, path: Path) -> tuple[float, ...]:
    value = _required(section, key, path)
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ValueError(f"'{key}' must be a sequence in {path}")
    return tuple(float(item) for item in value)
