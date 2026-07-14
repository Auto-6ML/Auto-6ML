from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import time

import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

from .config import find_dataset_config, load_dataset_config
from .data import DatasetBundle, load_dataset
from .graph_ops import build_feature_knn_graph, to_dense_float
from .losses import build_learned_graph, graph_structure_loss
from .model import MGBOModel
from .utils import as_index_tensor, set_seed


@dataclass
class TrainOptions:
    dataset: str = "acm"
    project_dir: Path = Path(".")
    data_dir: Path = Path("data")
    config_dir: Path = Path("configs")
    output_dir: Path = Path("checkpoints")
    seed: int = 1
    no_cuda: bool = False
    use_pretrain: bool = True
    eval_only: bool = False
    pretrain_dir: Path | None = None
    pretrain_epoch: int | None = None
    pretrain_model_path: Path | None = None
    pretrain_graph_path: Path | None = None


def _prepare_bundle(bundle: DatasetBundle, device: torch.device) -> DatasetBundle:
    return DatasetBundle(
        name=bundle.name,
        metapaths=[to_dense_float(metapath, device).requires_grad_(False) for metapath in bundle.metapaths],
        features=bundle.features.float().to(device),
        labels=bundle.labels.long().to(device),
        num_classes=bundle.num_classes,
        train_idx=as_index_tensor(bundle.train_idx, device),
        val_idx=as_index_tensor(bundle.val_idx, device),
        test_idx=as_index_tensor(bundle.test_idx, device),
    )


def _delete_old_checkpoints(output_dir: Path, prefix: str) -> None:
    for path in output_dir.glob(f"{prefix}_*.bestmodel.pt"):
        path.unlink()
    for path in output_dir.glob(f"{prefix}_*.bestgraph.pt"):
        path.unlink()


def _checkpoint_paths(output_dir: Path, prefix: str, epoch: int) -> tuple[Path, Path]:
    model_path = output_dir / f"{prefix}_{epoch}.bestmodel.pt"
    graph_path = output_dir / f"{prefix}_{epoch}.bestgraph.pt"
    return model_path, graph_path


def _published_checkpoint_paths(output_dir: Path, prefix: str) -> tuple[Path, Path]:
    model_path = output_dir / f"{prefix}.model.pt"
    graph_path = output_dir / f"{prefix}.graph.pt"
    return model_path, graph_path


def _resolve_path(project_dir: Path, path: Path) -> Path:
    return path if path.is_absolute() else project_dir / path


def _checkpoint_epoch(path: Path, prefix: str, suffix: str) -> int | None:
    match = re.fullmatch(rf"{re.escape(prefix)}_(\d+)\.{re.escape(suffix)}", path.name)
    return None if match is None else int(match.group(1))


def _find_checkpoint_pair(
    project_dir: Path,
    output_dir: Path,
    prefix: str,
    options: TrainOptions,
    default_dir: Path | None = None,
) -> tuple[Path, Path, int | None]:
    if options.pretrain_model_path is not None or options.pretrain_graph_path is not None:
        if options.pretrain_model_path is None or options.pretrain_graph_path is None:
            raise ValueError("--pretrain-model-path and --pretrain-graph-path must be provided together")
        model_path = _resolve_path(project_dir, options.pretrain_model_path)
        graph_path = _resolve_path(project_dir, options.pretrain_graph_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Cannot find pretrained model: {model_path}")
        if not graph_path.exists():
            raise FileNotFoundError(f"Cannot find pretrained graph: {graph_path}")
        return model_path, graph_path, None

    fallback_dir = output_dir if default_dir is None else _resolve_path(project_dir, default_dir)
    pretrain_dir = fallback_dir if options.pretrain_dir is None else _resolve_path(project_dir, options.pretrain_dir)
    model_path, graph_path = _published_checkpoint_paths(pretrain_dir, prefix)
    if model_path.exists() and graph_path.exists():
        return model_path, graph_path, None

    if options.pretrain_epoch is not None:
        model_path, graph_path = _checkpoint_paths(pretrain_dir, prefix, options.pretrain_epoch)
        if not model_path.exists():
            raise FileNotFoundError(f"Cannot find pretrained model: {model_path}")
        if not graph_path.exists():
            raise FileNotFoundError(f"Cannot find pretrained graph: {graph_path}")
        return model_path, graph_path, options.pretrain_epoch

    model_files = sorted(pretrain_dir.glob(f"{prefix}_*.bestmodel.pt"))
    graph_files = sorted(pretrain_dir.glob(f"{prefix}_*.bestgraph.pt"))
    model_epochs = {
        epoch: path
        for path in model_files
        if (epoch := _checkpoint_epoch(path, prefix, "bestmodel.pt")) is not None
    }
    graph_epochs = {
        epoch: path
        for path in graph_files
        if (epoch := _checkpoint_epoch(path, prefix, "bestgraph.pt")) is not None
    }
    common_epochs = sorted(set(model_epochs) & set(graph_epochs))
    if not common_epochs:
        raise FileNotFoundError(
            f"Cannot find pretrained checkpoint pair for '{prefix}' in {pretrain_dir}. "
            f"Expected files like {prefix}.model.pt and {prefix}.graph.pt."
        )

    if len(common_epochs) == 1:
        epoch = common_epochs[0]
    else:
        epoch = max(
            common_epochs,
            key=lambda item: min(model_epochs[item].stat().st_mtime, graph_epochs[item].stat().st_mtime),
        )
        print(
            f"Found {len(common_epochs)} checkpoint pairs for {prefix} in {pretrain_dir}; "
            f"using the most recently modified pair at epoch {epoch}. "
            "Use --pretrain-epoch to select a specific one."
        )
    return model_epochs[epoch], graph_epochs[epoch], epoch


def _evaluate_model(
    model: MGBOModel,
    bundle: DatasetBundle,
    learned_graph: torch.Tensor,
    topology_norm: int | float,
) -> tuple[float, float]:
    model.eval()
    with torch.no_grad():
        output, _ = model(bundle.features, learned_graph, bundle.metapaths, topology_norm)
        predictions = output[bundle.test_idx].argmax(dim=1).detach().cpu().numpy()
        labels = bundle.labels[bundle.test_idx].detach().cpu().numpy()
        macro_f1 = f1_score(labels, predictions, average="macro")
        micro_f1 = f1_score(labels, predictions, average="micro")
    return float(macro_f1), float(micro_f1)


def _manual_hyper_inner_update(
    C: torch.Tensor,
    C_optimizer: torch.optim.Optimizer,
    embeddings: torch.Tensor,
    metapaths: list[torch.Tensor],
    model: MGBOModel,
    features_dim: int,
    steps: int,
    reconstruction_weight: float,
    regularization_weight: float,
    metapath_weights: tuple[float, ...],
) -> tuple[torch.Tensor, torch.Tensor | None]:
    jacobian = torch.zeros_like(model.gcn.weight)
    graph_loss = None

    for _ in range(steps):
        C.requires_grad_(True)
        C_optimizer.zero_grad(set_to_none=True)
        graph_loss = graph_structure_loss(
            C,
            embeddings,
            metapaths,
            reconstruction_weight=reconstruction_weight,
            regularization_weight=regularization_weight,
            metapath_weights=metapath_weights,
        )
        grad_C = torch.autograd.grad(graph_loss, C, create_graph=True, retain_graph=True)[0]

        gcn_jacobian_term = torch.autograd.grad(
            grad_C.mean(),
            model.gcn.weight,
            retain_graph=True,
            allow_unused=True,
        )[0]
        inner_jacobian_term = torch.autograd.grad(grad_C.mean(), C, retain_graph=True)[0]

        if gcn_jacobian_term is not None:
            inner_scale = (1.0 - inner_jacobian_term).mean(dim=1).unsqueeze(1)
            inner_scale = inner_scale.expand(-1, features_dim)
            jacobian = inner_scale.mean(dim=0).unsqueeze(1) * jacobian + gcn_jacobian_term

        C.grad = grad_C.detach()
        C_optimizer.step()

    return jacobian, graph_loss


def _assign_manual_hyper_grads(
    model: MGBOModel,
    loss: torch.Tensor,
    C: torch.Tensor,
    jacobian: torch.Tensor,
    features_dim: int,
) -> None:
    params = list(model.parameters())
    direct_grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    c_grad = torch.autograd.grad(loss, C, retain_graph=True, allow_unused=True)[0]

    indirect_gcn_grad = torch.zeros_like(model.gcn.weight)
    if c_grad is not None:
        c_scale = c_grad.mean(dim=1).unsqueeze(1).expand(-1, features_dim)
        indirect_gcn_grad = c_scale.mean(dim=0).unsqueeze(1) * jacobian

    for param, grad in zip(params, direct_grads):
        param.grad = None if grad is None else grad.detach().clone()

    if model.gcn.weight.grad is None:
        model.gcn.weight.grad = indirect_gcn_grad.detach().clone()
    else:
        model.gcn.weight.grad = model.gcn.weight.grad + indirect_gcn_grad.detach()


def train(options: TrainOptions) -> dict[str, float]:
    project_dir = options.project_dir.resolve()
    data_dir = (project_dir / options.data_dir).resolve()
    config_dir = (project_dir / options.config_dir).resolve()
    output_dir = (project_dir / options.output_dir).resolve()
    if not options.eval_only:
        output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(options.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not options.no_cuda else "cpu")

    config_path = find_dataset_config(config_dir, options.dataset)
    dataset_config = load_dataset_config(config_path, options.dataset)
    data_config = dataset_config.data
    training_config = dataset_config.training
    bundle = _prepare_bundle(load_dataset(options.dataset, data_dir), device)

    if data_config.n != bundle.features.shape[0]:
        raise ValueError(f"Configured n={data_config.n} does not match {options.dataset} data")
    if data_config.fdim != bundle.features.shape[1]:
        raise ValueError(f"Configured fdim={data_config.fdim} does not match {options.dataset} data")
    if data_config.class_num != bundle.num_classes:
        raise ValueError(f"Configured class_num={data_config.class_num} does not match {options.dataset} data")

    hidden_size = training_config.hidden_size
    num_classes = data_config.class_num
    if int(bundle.labels.max().item()) >= num_classes:
        raise ValueError(f"num_classes={num_classes} is too small for labels in {options.dataset}")
    dropout = training_config.dropout
    lr = training_config.lr
    weight_decay = training_config.weight_decay
    knn_k = training_config.knn_k
    topology_norm = training_config.topology_norm

    model = MGBOModel(
        input_size=bundle.features.shape[1],
        hidden_size=hidden_size,
        num_classes=num_classes,
        num_metapaths=len(bundle.metapaths),
        dropout=dropout,
        graph_clamp_negative=training_config.graph_clamp_negative,
    ).to(device)

    prefix = options.dataset
    if options.eval_only:
        start = time.time()
        model_path, graph_path, pretrain_epoch = _find_checkpoint_pair(
            project_dir,
            output_dir,
            prefix,
            options,
            default_dir=Path("pretrained"),
        )
        model.load_state_dict(torch.load(model_path, map_location=device))
        learned_graph = torch.load(graph_path, map_location=device)
        macro_f1, micro_f1 = _evaluate_model(model, bundle, learned_graph, topology_norm)

        metrics = {
            "best_epoch": float(pretrain_epoch or 0),
            "best_score": float("nan"),
            "macro_f1": macro_f1,
            "micro_f1": micro_f1,
            "seconds": float(time.time() - start),
        }
        print(
            f"Loaded pretrained MGBO checkpoint: dataset={options.dataset}, "
            f"model={model_path}, graph={graph_path}, "
            f"device={device}, hidden_size={hidden_size}, num_classes={num_classes}, "
            f"topology_norm={topology_norm}, graph_clamp_negative={training_config.graph_clamp_negative}"
        )
        print(
            f"macro_f1={metrics['macro_f1']:.4f}",
            f"micro_f1={metrics['micro_f1']:.4f}",
            f"time: {metrics['seconds']:.4f}s",
        )
        return metrics

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    initial_graph = build_feature_knn_graph(bundle.features, k=knn_k).to(device).float()
    C = initial_graph.clone().detach().requires_grad_(True)
    C_optimizer = torch.optim.Adam([C], lr=training_config.inner_lr)

    epochs = training_config.epochs
    inner_steps = training_config.inner_steps
    best_score = float("-inf")
    best_epoch = 0
    stale_epochs = 0
    start = time.time()

    print(
        f"Begin MGBO bilevel training: dataset={options.dataset}, "
        f"config={config_path.name}, "
        f"device={device}, epochs={epochs}, inner_steps={inner_steps}, "
        f"inner_lr={training_config.inner_lr:g}, bilevel_mode=manual_hyper, "
        f"lr={lr:g}, weight_decay={weight_decay:g}, dropout={dropout:g}, "
        f"hidden_size={hidden_size}, num_classes={num_classes}, knn_k={knn_k}, topology_norm={topology_norm}, "
        f"learned_graph_norm={training_config.learned_graph_norm}, "
        f"checkpoint_metric=val_macro, graph_clamp_negative={training_config.graph_clamp_negative}, "
        f"use_pretrain={options.use_pretrain}"
    )

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        optimizer.zero_grad(set_to_none=True)

        seed_C = C.detach()
        seed_graph = build_learned_graph(
            seed_C,
            initial_graph,
            initial_graph_weight=training_config.initial_graph_weight,
            normalization=training_config.learned_graph_norm,
        )
        _, seed_embeddings = model(bundle.features, seed_graph, bundle.metapaths, topology_norm)

        structure_embeddings = bundle.features if epoch == 1 else seed_embeddings
        jacobian, graph_loss = _manual_hyper_inner_update(
            C=C,
            C_optimizer=C_optimizer,
            embeddings=structure_embeddings,
            metapaths=bundle.metapaths,
            model=model,
            features_dim=bundle.features.shape[1],
            steps=inner_steps,
            reconstruction_weight=training_config.reconstruction_weight,
            regularization_weight=training_config.regularization_weight,
            metapath_weights=training_config.metapath_weights,
        )
        learned_graph = build_learned_graph(
            C,
            initial_graph,
            initial_graph_weight=training_config.initial_graph_weight,
            normalization=training_config.learned_graph_norm,
        )
        output, _ = model(bundle.features, learned_graph, bundle.metapaths, topology_norm)

        train_loss = F.nll_loss(output[bundle.train_idx], bundle.labels[bundle.train_idx])
        val_loss = F.nll_loss(output[bundle.val_idx], bundle.labels[bundle.val_idx])
        val_predictions = output[bundle.val_idx].argmax(dim=1).detach().cpu().numpy()
        val_labels = bundle.labels[bundle.val_idx].detach().cpu().numpy()
        val_macro = f1_score(val_labels, val_predictions, average="macro", zero_division=0)
        val_micro = f1_score(val_labels, val_predictions, average="micro", zero_division=0)
        score = float(val_macro)

        if score > best_score:
            best_score = score
            best_epoch = epoch
            stale_epochs = 0
            _delete_old_checkpoints(output_dir, prefix)
            model_path, graph_path = _checkpoint_paths(output_dir, prefix, epoch)
            torch.save(model.state_dict(), model_path)
            torch.save(learned_graph.detach(), graph_path)
        else:
            stale_epochs += 1

        _assign_manual_hyper_grads(
            model=model,
            loss=train_loss,
            C=C,
            jacobian=jacobian,
            features_dim=bundle.features.shape[1],
        )
        optimizer.step()

        graph_loss_value = 0.0 if graph_loss is None else graph_loss.item()
        print(
            f"Epoch: {epoch:04d}",
            f"loss_graph: {graph_loss_value:.4f}",
            f"loss_train: {train_loss.item():.4f}",
            f"loss_val: {val_loss.item():.4f}",
            f"macro_val: {val_macro:.4f}",
            f"micro_val: {val_micro:.4f}",
            f"best_score: {best_score:.4f}",
            f"best_epoch: {best_epoch}",
            f"time: {time.time() - epoch_start:.4f}s",
        )

        if stale_epochs >= training_config.patience and epoch >= training_config.min_epochs:
            print(f"Validation metric has not improved for {training_config.patience} epochs, early stopping...")
            break

    if options.use_pretrain:
        model_path, graph_path, pretrain_epoch = _find_checkpoint_pair(
            project_dir,
            output_dir,
            prefix,
            options,
            default_dir=Path("pretrained"),
        )
        model.load_state_dict(torch.load(model_path, map_location=device))
        learned_graph = torch.load(graph_path, map_location=device)
        evaluated_epoch = float(pretrain_epoch or 0)
        evaluated_score = float("nan")
        print(f"Evaluating pretrained checkpoint after training: model={model_path}, graph={graph_path}")
    else:
        model_path, graph_path = _checkpoint_paths(output_dir, prefix, best_epoch)
        model.load_state_dict(torch.load(model_path, map_location=device))
        learned_graph = torch.load(graph_path, map_location=device)
        evaluated_epoch = float(best_epoch)
        evaluated_score = float(best_score)
    macro_f1, micro_f1 = _evaluate_model(model, bundle, learned_graph, topology_norm)

    metrics = {
        "best_epoch": evaluated_epoch,
        "best_score": evaluated_score,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "seconds": float(time.time() - start),
    }
    print(
        f"macro_f1={metrics['macro_f1']:.4f}",
        f"micro_f1={metrics['micro_f1']:.4f}",
        f"time: {metrics['seconds']:.4f}s",
    )
    return metrics
