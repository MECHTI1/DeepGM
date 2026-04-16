from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch_geometric.loader import DataLoader

from label_schemes import EC_TOP_LEVEL_LABELS, METAL_TARGET_LABELS, N_EC_CLASSES, N_METAL_CLASSES
from model import GVPPocketClassifier
from project_paths import resolve_runs_dir
from training.config import TrainConfig, config_to_payload
from training.data import load_training_pockets_with_report_from_dir
from training.graph_dataset import (
    FeatureNormalizationStats,
    PocketGraphDataset,
    build_graph_data_list,
    compute_feature_normalization_stats,
)
from training.loop import (
    balanced_class_weights_from_pockets,
    classification_metrics_from_logits,
    evaluate_epoch_with_predictions,
    train_epoch,
)
from training.preflight import run_preflight_checks
from training.splits import PocketSplit, build_dataset_summary, split_pockets


@dataclass(frozen=True)
class PreparedRun:
    config_payload: dict[str, Any]
    run_dir: Path
    split: PocketSplit
    dataset_summary: dict[str, Any]
    normalization_stats: FeatureNormalizationStats
    train_loader: DataLoader
    val_loader: DataLoader | None
    model: GVPPocketClassifier
    optimizer: torch.optim.Optimizer


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_run_dir(config: TrainConfig) -> Path:
    runs_dir = resolve_runs_dir(config.runs_dir, create=True)
    effective_name = config.run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = runs_dir / effective_name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def validate_training_configuration(config: TrainConfig) -> None:
    if config.val_fraction == 0.0 and config.selection_metric.startswith("val_"):
        raise ValueError(
            "Selection metric "
            f"{config.selection_metric!r} requires validation, but --val-fraction is 0. "
            "Either enable validation or choose a train-based metric such as 'train_loss'."
        )


def evaluate_split_metrics(model, loader: DataLoader | None, device: str, prefix: str) -> dict[str, Any]:
    if loader is None:
        return {}

    predictions = evaluate_epoch_with_predictions(model, loader, device=device)
    payload = {
        f"{prefix}_loss": predictions["loss"],
    }
    if "metal_y" in predictions:
        metal_metrics = classification_metrics_from_logits(predictions["metal_logits"], predictions["metal_y"])
        payload.update(
            {
                f"{prefix}_metal_acc": metal_metrics["accuracy"],
                f"{prefix}_metal_balanced_acc": metal_metrics["balanced_accuracy"],
                f"{prefix}_metal_macro_f1": metal_metrics["macro_f1"],
                f"{prefix}_metal_per_class_recall": {
                    label_name: metal_metrics["per_class_recall"][label_idx]
                    for label_idx, label_name in METAL_TARGET_LABELS.items()
                },
            }
        )
    else:
        payload[f"{prefix}_metal_acc"] = None
    if "ec_y" in predictions:
        ec_metrics = classification_metrics_from_logits(predictions["ec_logits"], predictions["ec_y"])
        payload.update(
            {
                f"{prefix}_ec_acc": ec_metrics["accuracy"],
                f"{prefix}_ec_balanced_acc": ec_metrics["balanced_accuracy"],
                f"{prefix}_ec_macro_f1": ec_metrics["macro_f1"],
                f"{prefix}_ec_per_class_recall": {
                    label_name: ec_metrics["per_class_recall"][label_idx]
                    for label_idx, label_name in EC_TOP_LEVEL_LABELS.items()
                },
            }
        )
    else:
        payload[f"{prefix}_ec_acc"] = None

    balanced_values = [
        value
        for value in (
            payload.get(f"{prefix}_metal_balanced_acc"),
            payload.get(f"{prefix}_ec_balanced_acc"),
        )
        if value is not None
    ]
    macro_f1_values = [
        value
        for value in (
            payload.get(f"{prefix}_metal_macro_f1"),
            payload.get(f"{prefix}_ec_macro_f1"),
        )
        if value is not None
    ]
    if balanced_values:
        payload[f"{prefix}_joint_balanced_acc"] = float(sum(balanced_values) / len(balanced_values))
    if macro_f1_values:
        payload[f"{prefix}_joint_macro_f1"] = float(sum(macro_f1_values) / len(macro_f1_values))
    return payload


def normalization_stats_payload(normalization_stats: FeatureNormalizationStats) -> dict[str, Any]:
    return {
        "means": normalization_stats.means,
        "stds": normalization_stats.stds,
        "clamp_value": normalization_stats.clamp_value,
    }


def checkpoint_payload(
    *,
    model_state_dict: dict[str, Any],
    optimizer_state_dict: dict[str, Any],
    history: list[dict[str, Any]],
    config_payload: dict[str, Any],
    normalization_stats: FeatureNormalizationStats,
    dataset_summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model_state_dict": model_state_dict,
        "optimizer_state_dict": optimizer_state_dict,
        "history": history,
        "config": config_payload,
        "metal_labels": METAL_TARGET_LABELS,
        "ec_labels": EC_TOP_LEVEL_LABELS,
        "normalization_stats": normalization_stats_payload(normalization_stats),
        "dataset_summary": dataset_summary,
    }


def format_epoch_log(record: dict[str, Any]) -> str:
    parts = [
        f"epoch={record['epoch']}",
        f"train_loss={record['train_loss']:.4f}",
    ]
    if record["train_metal_acc"] is not None:
        parts.append(f"train_metal_acc={record['train_metal_acc']:.4f}")
    if record["train_ec_acc"] is not None:
        parts.append(f"train_ec_acc={record['train_ec_acc']:.4f}")
    if "val_loss" in record:
        parts.append(f"val_loss={record['val_loss']:.4f}")
    if record.get("val_metal_acc") is not None:
        parts.append(f"val_metal_acc={record['val_metal_acc']:.4f}")
    if record.get("val_ec_acc") is not None:
        parts.append(f"val_ec_acc={record['val_ec_acc']:.4f}")
    if record.get("val_joint_balanced_acc") is not None:
        parts.append(f"val_joint_bal_acc={record['val_joint_balanced_acc']:.4f}")
    if record.get("val_joint_macro_f1") is not None:
        parts.append(f"val_joint_macro_f1={record['val_joint_macro_f1']:.4f}")
    return " ".join(parts)


def metric_sort_value(record: dict[str, Any], selection_metric: str) -> tuple[float, bool]:
    if selection_metric not in record or record[selection_metric] is None:
        raise ValueError(f"Selection metric {selection_metric!r} is missing from the epoch record.")
    metric_value = float(record[selection_metric])
    if selection_metric.endswith("_loss"):
        return metric_value, False
    return metric_value, True


def prepare_run(config: TrainConfig) -> PreparedRun:
    validate_training_configuration(config)
    config_payload = config_to_payload(config)
    load_result = load_training_pockets_with_report_from_dir(
        structure_dir=config.structure_dir,
        require_full_labels=True,
        summary_csv=config.summary_csv,
        esm_dim=config.esm_dim,
        esm_embeddings_dir=config.esm_embeddings_dir,
        require_esm_embeddings=config.require_esm_embeddings,
        external_features_root_dir=config.external_features_root_dir,
        require_external_features=config.require_external_features,
        unsupported_metal_policy=config.unsupported_metal_policy,
    )
    pockets = load_result.pockets
    if not pockets:
        raise ValueError("No training pockets were loaded.")

    split = split_pockets(
        pockets,
        val_fraction=config.val_fraction,
        split_by=config.split_by,
        seed=config.seed,
    )
    dataset_summary = build_dataset_summary(
        split,
        config,
        feature_load_report=load_result.feature_report,
    )
    train_graphs = build_graph_data_list(
        split.train_pockets,
        esm_dim=config.esm_dim,
        edge_radius=config.edge_radius,
        require_ring_edges=config.require_ring_edges,
    )
    val_graphs = (
        build_graph_data_list(
            split.val_pockets,
            esm_dim=config.esm_dim,
            edge_radius=config.edge_radius,
            require_ring_edges=config.require_ring_edges,
        )
        if split.val_pockets
        else None
    )
    dataset_summary["preflight"] = run_preflight_checks(
        split,
        config,
        train_graphs=train_graphs,
        val_graphs=val_graphs,
    )
    run_dir = build_run_dir(config)
    normalization_stats = compute_feature_normalization_stats(train_graphs, clamp_value=5.0)
    train_loader = DataLoader(
        PocketGraphDataset(
            split.train_pockets,
            esm_dim=config.esm_dim,
            edge_radius=config.edge_radius,
            normalization_stats=normalization_stats,
            require_ring_edges=config.require_ring_edges,
            precomputed_data=train_graphs,
        ),
        batch_size=config.batch_size,
        shuffle=True,
    )
    val_loader = (
        DataLoader(
            PocketGraphDataset(
                split.val_pockets,
                esm_dim=config.esm_dim,
                edge_radius=config.edge_radius,
                normalization_stats=normalization_stats,
                require_ring_edges=config.require_ring_edges,
                precomputed_data=val_graphs,
            ),
            batch_size=config.batch_size,
            shuffle=False,
        )
        if split.val_pockets
        else None
    )

    metal_class_weights, ec_class_weights = balanced_class_weights_from_pockets(
        split.train_pockets,
        n_metal_classes=N_METAL_CLASSES,
        n_ec_classes=N_EC_CLASSES,
    )

    model = GVPPocketClassifier(
        esm_dim=config.esm_dim,
        metal_class_weights=metal_class_weights,
        ec_class_weights=ec_class_weights,
    ).to(config.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    return PreparedRun(
        config_payload=config_payload,
        run_dir=run_dir,
        split=split,
        dataset_summary=dataset_summary,
        normalization_stats=normalization_stats,
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        optimizer=optimizer,
    )


def train_and_select_checkpoint(
    prepared: PreparedRun,
    config: TrainConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    history: list[dict[str, Any]] = []
    best_metric: float | None = None
    best_checkpoint = None
    for epoch in range(1, config.epochs + 1):
        train_loss = train_epoch(prepared.model, prepared.train_loader, prepared.optimizer, device=config.device)
        train_metrics = evaluate_split_metrics(prepared.model, prepared.train_loader, config.device, prefix="train")
        train_metrics.pop("train_loss", None)
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            **train_metrics,
        }

        val_metrics = evaluate_split_metrics(prepared.model, prepared.val_loader, config.device, prefix="val")
        record.update(val_metrics)
        current_metric, maximize = metric_sort_value(record, config.selection_metric)
        is_better = (
            best_metric is None
            or (maximize and current_metric > best_metric)
            or (not maximize and current_metric < best_metric)
        )
        if is_better:
            best_metric = current_metric
            best_checkpoint = checkpoint_payload(
                model_state_dict=copy.deepcopy(prepared.model.state_dict()),
                optimizer_state_dict=copy.deepcopy(prepared.optimizer.state_dict()),
                history=copy.deepcopy(history + [record]),
                config_payload=prepared.config_payload,
                normalization_stats=prepared.normalization_stats,
                dataset_summary=prepared.dataset_summary,
            )
            best_checkpoint["epoch"] = epoch
            best_checkpoint["selection_metric"] = config.selection_metric
            best_checkpoint["selection_metric_value"] = current_metric

        history.append(record)
        print(format_epoch_log(record))
    return history, best_checkpoint


def persist_run_outputs(
    prepared: PreparedRun,
    *,
    history: list[dict[str, float | int]],
    best_checkpoint: dict[str, Any] | None,
) -> None:
    save_json(prepared.run_dir / "dataset_summary.json", prepared.dataset_summary)

    checkpoint_path = prepared.run_dir / "last_model_checkpoint.pt"
    torch.save(
        checkpoint_payload(
            model_state_dict=prepared.model.state_dict(),
            optimizer_state_dict=prepared.optimizer.state_dict(),
            history=history,
            config_payload=prepared.config_payload,
            normalization_stats=prepared.normalization_stats,
            dataset_summary=prepared.dataset_summary,
        ),
        checkpoint_path,
    )

    if best_checkpoint is not None:
        best_checkpoint_path = prepared.run_dir / "best_model_checkpoint.pt"
        torch.save(best_checkpoint, best_checkpoint_path)

    save_json(
        prepared.run_dir / "run_config.json",
        {
            "config": prepared.config_payload,
            "dataset_summary": prepared.dataset_summary,
            "metal_labels": METAL_TARGET_LABELS,
            "ec_labels": EC_TOP_LEVEL_LABELS,
            "history": history,
        },
    )

    print(f"Saved checkpoint to {checkpoint_path}")
    if best_checkpoint is not None:
        print(f"Saved best checkpoint to {prepared.run_dir / 'best_model_checkpoint.pt'}")
    print(f"Saved dataset summary to {prepared.run_dir / 'dataset_summary.json'}")
    print(f"Saved run config to {prepared.run_dir / 'run_config.json'}")


def run_training(config: TrainConfig) -> Path:
    set_seed(config.seed)
    prepared = prepare_run(config)
    history, best_checkpoint = train_and_select_checkpoint(prepared, config)
    persist_run_outputs(prepared, history=history, best_checkpoint=best_checkpoint)
    return prepared.run_dir
