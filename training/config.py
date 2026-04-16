from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from data_structures import DEFAULT_EDGE_RADIUS
from training.data import DEFAULT_STRUCTURE_DIR, DEFAULT_TRAIN_SUMMARY_CSV
from training.esm_feature_loading import DEFAULT_ESMC_EMBED_DIM

VALID_SPLIT_BY_CHOICES = ("pdbid", "pdbid_chain", "structure_id", "pocket_id")
VALID_UNSUPPORTED_METAL_POLICY_CHOICES = ("error", "skip")
VALID_SELECTION_METRIC_CHOICES = (
    "train_loss",
    "val_loss",
    "val_joint_balanced_acc",
    "val_joint_macro_f1",
    "val_metal_balanced_acc",
    "val_ec_balanced_acc",
    "val_metal_macro_f1",
    "val_ec_macro_f1",
)


@dataclass(frozen=True)
class TrainConfig:
    structure_dir: Path = DEFAULT_STRUCTURE_DIR
    summary_csv: Path = DEFAULT_TRAIN_SUMMARY_CSV
    esm_embeddings_dir: str | None = None
    external_features_root_dir: str | None = None
    runs_dir: str | None = None
    run_name: str | None = None
    device: str = "cpu"
    epochs: int = 10
    batch_size: int = 8
    learning_rate: float = 3e-4
    # `0.0` is useful for smoke runs; real training should usually use a nonzero validation split.
    val_fraction: float = 0.0
    esm_dim: int = DEFAULT_ESMC_EMBED_DIM
    edge_radius: float = DEFAULT_EDGE_RADIUS
    weight_decay: float = 1e-4
    seed: int = 42
    require_ring_edges: bool = False
    split_by: str = "pdbid"
    require_esm_embeddings: bool = True
    require_external_features: bool = True
    unsupported_metal_policy: str = "error"
    selection_metric: str = "train_loss"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the pocket classifier on the catalytic-only MAHOMES summary table."
    )
    parser.add_argument("--structure-dir", type=Path, default=DEFAULT_STRUCTURE_DIR)
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_TRAIN_SUMMARY_CSV)
    parser.add_argument("--esm-embeddings-dir", type=str, default=None)
    parser.add_argument("--external-features-root-dir", type=str, default=None)
    parser.add_argument("--runs-dir", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--esm-dim", type=int, default=DEFAULT_ESMC_EMBED_DIM)
    parser.add_argument("--edge-radius", type=float, default=DEFAULT_EDGE_RADIUS)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--require-ring-edges", action="store_true")
    parser.add_argument("--allow-missing-esm-embeddings", action="store_true")
    parser.add_argument("--allow-missing-external-features", action="store_true")
    parser.add_argument("--val-fraction", type=float, default=0.0)
    parser.add_argument(
        "--unsupported-metal-policy",
        type=str,
        default="error",
        choices=VALID_UNSUPPORTED_METAL_POLICY_CHOICES,
    )
    parser.add_argument(
        "--selection-metric",
        type=str,
        default=None,
        choices=VALID_SELECTION_METRIC_CHOICES,
    )
    parser.add_argument(
        "--split-by",
        type=str,
        default="pdbid",
        choices=VALID_SPLIT_BY_CHOICES,
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> TrainConfig:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    selection_metric = args.selection_metric
    if selection_metric is None:
        selection_metric = "val_joint_balanced_acc" if args.val_fraction > 0.0 else "train_loss"
    return TrainConfig(
        structure_dir=args.structure_dir,
        summary_csv=args.summary_csv,
        esm_embeddings_dir=args.esm_embeddings_dir,
        external_features_root_dir=args.external_features_root_dir,
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
        require_esm_embeddings=not args.allow_missing_esm_embeddings,
        require_external_features=not args.allow_missing_external_features,
        unsupported_metal_policy=args.unsupported_metal_policy,
        selection_metric=selection_metric,
    )


def config_to_payload(config: TrainConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["structure_dir"] = str(config.structure_dir)
    payload["summary_csv"] = str(config.summary_csv)
    return payload
