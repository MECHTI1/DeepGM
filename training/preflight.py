from __future__ import annotations

from typing import Any

from data_structures import PocketRecord
from graph.construction import pocket_to_pyg_data
from label_schemes import EC_TOP_LEVEL_LABELS, METAL_TARGET_LABELS
from training.config import TrainConfig
from training.feature_sources import build_pocket_feature_coverage
from training.splits import PocketSplit, pocket_split_key


def validate_graphs(
    pockets: list[PocketRecord],
    config: TrainConfig,
    precomputed_graphs: list[Any] | None = None,
) -> None:
    if precomputed_graphs is not None:
        if len(precomputed_graphs) != len(pockets):
            raise ValueError("Graph preflight received mismatched precomputed data.")
        return

    for pocket in pockets:
        try:
            pocket_to_pyg_data(
                pocket,
                esm_dim=config.esm_dim,
                edge_radius=config.edge_radius,
                require_ring_edges=config.require_ring_edges,
            )
        except Exception as exc:
            raise ValueError(f"Graph preflight failed for pocket {pocket.pocket_id!r}: {exc}") from exc


def run_preflight_checks(
    split: PocketSplit,
    config: TrainConfig,
    *,
    train_graphs: list[Any] | None = None,
    val_graphs: list[Any] | None = None,
) -> dict[str, object]:
    if not split.train_pockets:
        raise ValueError("Preflight failed: training split is empty.")
    if config.val_fraction > 0.0 and not split.val_pockets:
        raise ValueError("Preflight failed: validation split is empty, but --val-fraction > 0.")
    if config.val_fraction == 0.0 and split.val_pockets:
        raise ValueError("Preflight failed: validation pockets exist, but --val-fraction is 0.")

    empty_train = [pocket.pocket_id for pocket in split.train_pockets if not pocket.residues]
    if empty_train:
        raise ValueError(f"Preflight failed: training pockets without residues: {empty_train[:5]}")

    empty_val = [pocket.pocket_id for pocket in split.val_pockets if not pocket.residues]
    if empty_val:
        raise ValueError(f"Preflight failed: validation pockets without residues: {empty_val[:5]}")

    train_metal_ids = {
        int(pocket.y_metal) for pocket in split.train_pockets if pocket.y_metal is not None and int(pocket.y_metal) in METAL_TARGET_LABELS
    }
    train_ec_ids = {int(pocket.y_ec) for pocket in split.train_pockets if pocket.y_ec is not None and int(pocket.y_ec) in EC_TOP_LEVEL_LABELS}
    val_metal_ids = {int(pocket.y_metal) for pocket in split.val_pockets if pocket.y_metal is not None and int(pocket.y_metal) in METAL_TARGET_LABELS}
    val_ec_ids = {int(pocket.y_ec) for pocket in split.val_pockets if pocket.y_ec is not None and int(pocket.y_ec) in EC_TOP_LEVEL_LABELS}

    if len(train_metal_ids) < 2:
        raise ValueError("Preflight failed: training split contains fewer than 2 metal classes.")
    if len(train_ec_ids) < 2:
        raise ValueError("Preflight failed: training split contains fewer than 2 EC classes.")

    overlap = sorted(
        {pocket_split_key(pocket, config.split_by) for pocket in split.train_pockets}.intersection(
            pocket_split_key(pocket, config.split_by) for pocket in split.val_pockets
        )
    )
    if overlap:
        raise ValueError(
            "Preflight failed: train/validation leakage detected under "
            f"--split-by {config.split_by!r}: {overlap[:5]}"
        )

    validate_graphs(split.train_pockets, config, precomputed_graphs=train_graphs)
    validate_graphs(split.val_pockets, config, precomputed_graphs=val_graphs)

    train_feature_coverage = build_pocket_feature_coverage(split.train_pockets)
    val_feature_coverage = build_pocket_feature_coverage(split.val_pockets)
    warnings: list[str] = []

    if config.val_fraction > 0.0 and len(val_metal_ids) < 2:
        warnings.append("Validation split contains fewer than 2 metal classes.")
    if config.val_fraction > 0.0 and len(val_ec_ids) < 2:
        warnings.append("Validation split contains fewer than 2 EC classes.")
    if train_feature_coverage["esm_residue_coverage"] == 0.0:
        warnings.append("Training split has no ESM residue coverage.")
    if train_feature_coverage["external_feature_residue_coverage"] == 0.0:
        warnings.append("Training split has no external feature residue coverage.")
    if config.val_fraction > 0.0 and split.val_pockets and val_feature_coverage["esm_residue_coverage"] == 0.0:
        warnings.append("Validation split has no ESM residue coverage.")
    if (
        config.val_fraction > 0.0
        and split.val_pockets
        and val_feature_coverage["external_feature_residue_coverage"] == 0.0
    ):
        warnings.append("Validation split has no external feature residue coverage.")

    return {"warnings": warnings}
