from __future__ import annotations

import argparse
import json
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
    predict_batch,
    train_epoch,
)
from training_data import DEFAULT_STRUCTURE_DIR, DEFAULT_TRAIN_SUMMARY_CSV, load_training_pockets_from_dir


def parse_args() -> argparse.Namespace:
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
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_run_dir(runs_dir_arg: str | None, run_name: str | None) -> Path:
    runs_dir = resolve_runs_dir(runs_dir_arg, create=True)
    effective_name = run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = runs_dir / effective_name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    run_dir = build_run_dir(args.runs_dir, args.run_name)
    pockets = load_training_pockets_from_dir(
        structure_dir=args.structure_dir,
        require_full_labels=True,
        summary_csv=args.summary_csv,
    )
    if not pockets:
        raise ValueError("No training pockets were loaded.")

    normalization_stats = PocketGraphDataset.fit_normalization_stats(
        pockets,
        esm_dim=args.esm_dim,
        edge_radius=args.edge_radius,
        clamp_value=5.0,
        require_ring_edges=args.require_ring_edges,
    )
    dataset = PocketGraphDataset(
        pockets,
        esm_dim=args.esm_dim,
        edge_radius=args.edge_radius,
        normalization_stats=normalization_stats,
        require_ring_edges=args.require_ring_edges,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    metal_class_weights, ec_class_weights = balanced_class_weights_from_pockets(
        pockets,
        n_metal_classes=N_METAL_CLASSES,
        n_ec_classes=N_EC_CLASSES,
    )

    model = GVPPocketClassifier(
        esm_dim=args.esm_dim,
        metal_class_weights=metal_class_weights,
        ec_class_weights=ec_class_weights,
    ).to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    history: list[dict[str, float | int]] = []
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, loader, optimizer, device=args.device)
        predictions = predict_batch(model, loader, device=args.device)
        metal_acc = accuracy_from_logits(predictions["metal_logits"], predictions["metal_y"])
        ec_acc = accuracy_from_logits(predictions["ec_logits"], predictions["ec_y"])
        record = {
            "epoch": epoch,
            "loss": loss,
            "metal_acc": metal_acc,
            "ec_acc": ec_acc,
        }
        history.append(record)
        print(
            f"epoch={epoch} loss={loss:.4f} metal_acc={metal_acc:.4f} ec_acc={ec_acc:.4f}"
        )

    checkpoint_path = run_dir / "model_checkpoint.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
            "args": vars(args),
            "metal_labels": METAL_TARGET_LABELS,
            "ec_labels": EC_TOP_LEVEL_LABELS,
            "normalization_stats": {
                "means": normalization_stats.means,
                "stds": normalization_stats.stds,
                "clamp_value": normalization_stats.clamp_value,
            },
            "n_pockets": len(pockets),
        },
        checkpoint_path,
    )

    save_json(
        run_dir / "run_config.json",
        {
            "args": vars(args),
            "n_pockets": len(pockets),
            "metal_labels": METAL_TARGET_LABELS,
            "ec_labels": EC_TOP_LEVEL_LABELS,
            "history": history,
        },
    )

    print(f"Saved checkpoint to {checkpoint_path}")
    print(f"Saved run config to {run_dir / 'run_config.json'}")


if __name__ == "__main__":
    main()
