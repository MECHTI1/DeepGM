from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch_geometric.loader import DataLoader

from label_schemes import EC_TOP_LEVEL_LABELS, METAL_TARGET_LABELS, N_EC_CLASSES, N_METAL_CLASSES
from model import GVPPocketClassifier
from project_paths import resolve_runs_dir
from train_utils import (
    PocketGraphDataset,
    accuracy_from_logits,
    balanced_class_weights_from_pockets,
    evaluate_epoch,
    predict_batch,
    train_epoch,
)
from training_data import DEFAULT_STRUCTURE_DIR, DEFAULT_TRAIN_SUMMARY_CSV, load_training_pockets_from_dir
from training_labels import parse_structure_identity

SPLIT_BY_CHOICES = ("pdbid", "structure_id", "pocket_id")


@dataclass(frozen=True)
class TrainConfig:
    structure_dir: Path = DEFAULT_STRUCTURE_DIR
    summary_csv: Path = DEFAULT_TRAIN_SUMMARY_CSV
    runs_dir: str | None = None
    run_name: str | None = None
    device: str = "cpu"
    epochs: int = 10
    batch_size: int = 8
    learning_rate: float = 3e-4
    val_fraction: float = 0.0
    esm_dim: int = 256
    edge_radius: float = 10.0
    weight_decay: float = 1e-4
    seed: int = 42
    require_ring_edges: bool = False
    split_by: str = "pdbid"


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(
        description="Train the pocket classifier on the catalytic-only MAHOMES summary table."
    )
    parser.add_argument("--structure-dir", type=Path, default=DEFAULT_STRUCTURE_DIR)
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_TRAIN_SUMMARY_CSV)
    parser.add_argument("--runs-dir", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--esm-dim", type=int, default=256)
    parser.add_argument("--edge-radius", type=float, default=10.0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--require-ring-edges", action="store_true")
    parser.add_argument("--val-fraction", type=float, default=0.0)
    parser.add_argument(
        "--split-by",
        type=str,
        default="pdbid",
        choices=SPLIT_BY_CHOICES,
    )
    args = parser.parse_args()
    return TrainConfig(
        structure_dir=args.structure_dir,
        summary_csv=args.summary_csv,
        runs_dir=args.runs_dir,
        run_name=args.run_name,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        esm_dim=args.esm_dim,
        edge_radius=args.edge_radius,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        require_ring_edges=args.require_ring_edges,
        val_fraction=args.val_fraction,
        split_by=args.split_by,
    )


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


def config_to_payload(config: TrainConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["structure_dir"] = str(config.structure_dir)
    payload["summary_csv"] = str(config.summary_csv)
    return payload


def validate_split_by(split_by: str) -> str:
    if split_by not in SPLIT_BY_CHOICES:
        raise ValueError(
            f"Unsupported --split-by value: {split_by!r}. "
            "Expected one of: 'pdbid', 'structure_id', 'pocket_id'."
        )
    return split_by


def pocket_split_key(pocket, split_by: str) -> str:
    if split_by == "structure_id":
        return pocket.structure_id
    if split_by == "pdbid":
        pdbid, _chain, _ec = parse_structure_identity(pocket.structure_id)
        return pdbid
    if split_by == "pocket_id":
        return pocket.pocket_id
    raise AssertionError(f"Unhandled split_by value: {split_by!r}")


def split_pockets(pockets, val_fraction: float, split_by: str, seed: int):
    split_by = validate_split_by(split_by)
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError(f"--val-fraction must be in [0, 1), got {val_fraction}")
    if val_fraction == 0.0:
        return pockets, []

    grouped: dict[str, list] = {}
    for pocket in pockets:
        grouped.setdefault(pocket_split_key(pocket, split_by), []).append(pocket)

    group_items = list(grouped.items())
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(group_items), generator=generator).tolist()
    shuffled = [group_items[idx] for idx in order]

    target_val_size = max(1, int(round(len(pockets) * val_fraction)))
    val_pockets = []
    train_pockets = []
    val_count = 0

    for _group_key, group_pockets in shuffled:
        if val_count < target_val_size:
            val_pockets.extend(group_pockets)
            val_count += len(group_pockets)
        else:
            train_pockets.extend(group_pockets)

    if not train_pockets or not val_pockets:
        raise ValueError(
            "Validation split produced an empty train or validation set. "
            "Adjust --val-fraction or --split-by."
        )

    return train_pockets, val_pockets


def count_labels(pockets, attr_name: str, label_map: dict[int, str]) -> dict[str, int]:
    counts = {label_name: 0 for label_name in label_map.values()}
    for pocket in pockets:
        label_idx = getattr(pocket, attr_name)
        if label_idx is None:
            continue
        counts[label_map[int(label_idx)]] += 1
    return counts


def build_dataset_summary(train_pockets, val_pockets, config: TrainConfig) -> dict[str, Any]:
    return {
        "structure_dir": str(config.structure_dir),
        "summary_csv": str(config.summary_csv),
        "n_train_pockets": len(train_pockets),
        "n_val_pockets": len(val_pockets),
        "val_fraction": config.val_fraction,
        "split_by": config.split_by,
        "train_metal_distribution": count_labels(train_pockets, "y_metal", METAL_TARGET_LABELS),
        "train_ec_distribution": count_labels(train_pockets, "y_ec", EC_TOP_LEVEL_LABELS),
        "val_metal_distribution": count_labels(val_pockets, "y_metal", METAL_TARGET_LABELS),
        "val_ec_distribution": count_labels(val_pockets, "y_ec", EC_TOP_LEVEL_LABELS),
    }


def accuracy_or_none(predictions: dict[str, Any], logits_key: str, target_key: str) -> float | None:
    if target_key not in predictions:
        return None
    return accuracy_from_logits(predictions[logits_key], predictions[target_key])


def build_dataloader(pockets, config: TrainConfig, normalization_stats, shuffle: bool) -> DataLoader:
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

    loss = evaluate_epoch(model, loader, device=device)
    predictions = predict_batch(model, loader, device=device)
    return {
        f"{prefix}_loss": loss,
        f"{prefix}_metal_acc": accuracy_or_none(predictions, "metal_logits", "metal_y"),
        f"{prefix}_ec_acc": accuracy_or_none(predictions, "ec_logits", "ec_y"),
    }


def normalization_stats_payload(normalization_stats) -> dict[str, Any]:
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
    normalization_stats,
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


def main() -> None:
    config = parse_args()
    set_seed(config.seed)
    config_payload = config_to_payload(config)

    pockets = load_training_pockets_from_dir(
        structure_dir=config.structure_dir,
        require_full_labels=True,
        summary_csv=config.summary_csv,
    )
    if not pockets:
        raise ValueError("No training pockets were loaded.")

    train_pockets, val_pockets = split_pockets(
        pockets,
        val_fraction=config.val_fraction,
        split_by=config.split_by,
        seed=config.seed,
    )
    run_dir = build_run_dir(config)
    dataset_summary = build_dataset_summary(train_pockets, val_pockets, config)
    save_json(run_dir / "dataset_summary.json", dataset_summary)

    normalization_stats = PocketGraphDataset.fit_normalization_stats(
        train_pockets,
        esm_dim=config.esm_dim,
        edge_radius=config.edge_radius,
        clamp_value=5.0,
        require_ring_edges=config.require_ring_edges,
    )
    train_loader = build_dataloader(train_pockets, config, normalization_stats, shuffle=True)
    val_loader = (
        build_dataloader(val_pockets, config, normalization_stats, shuffle=False)
        if val_pockets
        else None
    )

    metal_class_weights, ec_class_weights = balanced_class_weights_from_pockets(
        train_pockets,
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

    history: list[dict[str, float | int]] = []
    best_metric = float("inf")
    best_checkpoint = None
    for epoch in range(1, config.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device=config.device)
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            **evaluate_split_metrics(model, train_loader, config.device, prefix="train"),
        }

        val_metrics = evaluate_split_metrics(model, val_loader, config.device, prefix="val")
        record.update(val_metrics)
        if val_metrics and val_metrics["val_loss"] < best_metric:
            best_metric = val_metrics["val_loss"]
            best_checkpoint = checkpoint_payload(
                model_state_dict=copy.deepcopy(model.state_dict()),
                optimizer_state_dict=copy.deepcopy(optimizer.state_dict()),
                history=copy.deepcopy(history + [record]),
                config_payload=config_payload,
                normalization_stats=normalization_stats,
                dataset_summary=dataset_summary,
            )
            best_checkpoint["epoch"] = epoch

        history.append(record)
        print(format_epoch_log(record))

    checkpoint_path = run_dir / "last_model_checkpoint.pt"
    torch.save(
        checkpoint_payload(
            model_state_dict=model.state_dict(),
            optimizer_state_dict=optimizer.state_dict(),
            history=history,
            config_payload=config_payload,
            normalization_stats=normalization_stats,
            dataset_summary=dataset_summary,
        ),
        checkpoint_path,
    )

    if best_checkpoint is not None:
        best_checkpoint_path = run_dir / "best_model_checkpoint.pt"
        torch.save(best_checkpoint, best_checkpoint_path)

    save_json(
        run_dir / "run_config.json",
        {
            "config": config_payload,
            "dataset_summary": dataset_summary,
            "metal_labels": METAL_TARGET_LABELS,
            "ec_labels": EC_TOP_LEVEL_LABELS,
            "history": history,
        },
    )

    print(f"Saved checkpoint to {checkpoint_path}")
    if best_checkpoint is not None:
        print(f"Saved best checkpoint to {run_dir / 'best_model_checkpoint.pt'}")
    print(f"Saved dataset summary to {run_dir / 'dataset_summary.json'}")
    print(f"Saved run config to {run_dir / 'run_config.json'}")


if __name__ == "__main__":
    main()
