from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch_geometric.loader import DataLoader

from data_structures import PocketRecord
from label_schemes import EC_TOP_LEVEL_LABELS, METAL_TARGET_LABELS, N_EC_CLASSES, N_METAL_CLASSES
from model import GVPPocketClassifier
from project_paths import resolve_runs_dir
from training.config import TrainConfig, config_to_payload
from training.data import load_training_pockets_with_report_from_dir
from training.graph_dataset import FeatureNormalizationStats, PocketGraphDataset
from training.loop import (
    accuracy_from_logits,
    balanced_class_weights_from_pockets,
    evaluate_epoch_with_predictions,
    train_epoch,
)
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


def accuracy_or_none(predictions: dict[str, Any], logits_key: str, target_key: str) -> float | None:
    if target_key not in predictions:
        return None
    return accuracy_from_logits(predictions[logits_key], predictions[target_key])


def build_dataloader(
    pockets: list[PocketRecord],
    config: TrainConfig,
    normalization_stats: FeatureNormalizationStats,
    shuffle: bool,
) -> DataLoader:
    dataset = PocketGraphDataset(
        pockets,
        esm_dim=config.esm_dim,
        edge_radius=config.edge_radius,
        normalization_stats=normalization_stats,
        require_ring_edges=config.require_ring_edges,
    )
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=shuffle)


def evaluate_split_metrics(model, loader: DataLoader | None, device: str, prefix: str) -> dict[str, Any]:
    if loader is None:
        return {}

    predictions = evaluate_epoch_with_predictions(model, loader, device=device)
    return {
        f"{prefix}_loss": predictions["loss"],
        f"{prefix}_metal_acc": accuracy_or_none(predictions, "metal_logits", "metal_y"),
        f"{prefix}_ec_acc": accuracy_or_none(predictions, "ec_logits", "ec_y"),
    }


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
    return " ".join(parts)


def prepare_run(config: TrainConfig) -> PreparedRun:
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
    run_dir = build_run_dir(config)
    dataset_summary = build_dataset_summary(
        split,
        config,
        feature_load_report=load_result.feature_report,
    )
    normalization_stats = PocketGraphDataset.fit_normalization_stats(
        split.train_pockets,
        esm_dim=config.esm_dim,
        edge_radius=config.edge_radius,
        clamp_value=5.0,
        require_ring_edges=config.require_ring_edges,
    )
    train_loader = build_dataloader(split.train_pockets, config, normalization_stats, shuffle=True)
    val_loader = (
        build_dataloader(split.val_pockets, config, normalization_stats, shuffle=False)
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
    best_metric = float("inf")
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
        if val_metrics and val_metrics["val_loss"] < best_metric:
            best_metric = val_metrics["val_loss"]
            best_checkpoint = checkpoint_payload(
                model_state_dict=copy.deepcopy(prepared.model.state_dict()),
                optimizer_state_dict=copy.deepcopy(prepared.optimizer.state_dict()),
                history=copy.deepcopy(history + [record]),
                config_payload=prepared.config_payload,
                normalization_stats=prepared.normalization_stats,
                dataset_summary=prepared.dataset_summary,
            )
            best_checkpoint["epoch"] = epoch

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


__all__ = [
    "PocketSplit",
    "accuracy_or_none",
    "build_dataset_summary",
    "build_dataloader",
    "build_run_dir",
    "checkpoint_payload",
    "evaluate_split_metrics",
    "format_epoch_log",
    "persist_run_outputs",
    "prepare_run",
    "run_training",
    "save_json",
    "set_seed",
    "split_pockets",
    "train_and_select_checkpoint",
]
