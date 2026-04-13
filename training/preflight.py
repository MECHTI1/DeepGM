from __future__ import annotations

from typing import Any

from data_structures import PocketRecord
from graph.construction import pocket_to_pyg_data
from label_schemes import EC_TOP_LEVEL_LABELS, METAL_TARGET_LABELS
from training.config import TrainConfig
from training.feature_sources import build_pocket_feature_coverage
from training.splits import PocketSplit, pocket_split_key


def label_coverage_summary(
    pockets: list[PocketRecord],
    attr_name: str,
    label_map: dict[int, str],
) -> dict[str, Any]:
    present_labels: list[str] = []
    missing_labels: list[str] = []
    n_labeled_pockets = 0

    seen_ids = set()
    for pocket in pockets:
        label_idx = getattr(pocket, attr_name)
        if label_idx is None:
            continue
        n_labeled_pockets += 1
        seen_ids.add(int(label_idx))

    for label_idx, label_name in label_map.items():
        if label_idx in seen_ids:
            present_labels.append(label_name)
        else:
            missing_labels.append(label_name)

    return {
        "n_labeled_pockets": n_labeled_pockets,
        "present_labels": present_labels,
        "missing_labels": missing_labels,
    }


def find_pockets_without_residues(pockets: list[PocketRecord]) -> list[str]:
    return [pocket.pocket_id for pocket in pockets if not pocket.residues]


def build_split_leakage_report(split: PocketSplit, split_by: str) -> dict[str, Any]:
    train_keys = {pocket_split_key(pocket, split_by) for pocket in split.train_pockets}
    val_keys = {pocket_split_key(pocket, split_by) for pocket in split.val_pockets}
    overlap = sorted(train_keys.intersection(val_keys))
    return {
        "split_by": split_by,
        "train_group_count": len(train_keys),
        "val_group_count": len(val_keys),
        "overlap_count": len(overlap),
        "overlap_examples": overlap[:5],
    }


def probe_graph_construction(
    pockets: list[PocketRecord],
    config: TrainConfig,
    sample_size: int = 3,
) -> dict[str, Any]:
    sample: list[dict[str, Any]] = []

    for pocket in pockets:
        try:
            data = pocket_to_pyg_data(
                pocket,
                esm_dim=config.esm_dim,
                edge_radius=config.edge_radius,
                require_ring_edges=config.require_ring_edges,
            )
        except Exception as exc:
            raise ValueError(f"Graph preflight failed for pocket {pocket.pocket_id!r}: {exc}") from exc

        if len(sample) < sample_size:
            sample.append(
                {
                    "pocket_id": pocket.pocket_id,
                    "n_nodes": int(data.pos.size(0)),
                    "n_edges": int(data.edge_index.size(1)),
                }
            )

    return {
        "checked_pocket_count": len(pockets),
        "sample": sample,
    }


def build_preflight_report(
    split: PocketSplit,
    config: TrainConfig,
    *,
    feature_load_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    warnings: list[str] = []

    train_metal_coverage = label_coverage_summary(split.train_pockets, "y_metal", METAL_TARGET_LABELS)
    train_ec_coverage = label_coverage_summary(split.train_pockets, "y_ec", EC_TOP_LEVEL_LABELS)
    val_metal_coverage = label_coverage_summary(split.val_pockets, "y_metal", METAL_TARGET_LABELS)
    val_ec_coverage = label_coverage_summary(split.val_pockets, "y_ec", EC_TOP_LEVEL_LABELS)
    train_feature_coverage = build_pocket_feature_coverage(split.train_pockets)
    val_feature_coverage = build_pocket_feature_coverage(split.val_pockets)
    leakage_check = build_split_leakage_report(split, config.split_by)

    if config.val_fraction > 0.0 and not split.val_pockets:
        warnings.append("Validation was requested but the validation split is empty.")
    if config.val_fraction > 0.0 and len(val_metal_coverage["present_labels"]) < 2:
        warnings.append("Validation split covers fewer than 2 metal classes.")
    if config.val_fraction > 0.0 and len(val_ec_coverage["present_labels"]) < 2:
        warnings.append("Validation split covers fewer than 2 EC classes.")
    if leakage_check["overlap_count"] > 0:
        warnings.append("Train/validation leakage was detected for the selected split key.")
    if train_feature_coverage["esm_residue_coverage"] == 0.0:
        warnings.append("Train split has zero ESM residue coverage.")
    if train_feature_coverage["external_feature_residue_coverage"] == 0.0:
        warnings.append("Train split has zero external-feature residue coverage.")
    if config.val_fraction > 0.0 and split.val_pockets and val_feature_coverage["esm_residue_coverage"] == 0.0:
        warnings.append("Validation split has zero ESM residue coverage.")
    if (
        config.val_fraction > 0.0
        and split.val_pockets
        and val_feature_coverage["external_feature_residue_coverage"] == 0.0
    ):
        warnings.append("Validation split has zero external-feature residue coverage.")

    return {
        "train_pocket_count": len(split.train_pockets),
        "val_pocket_count": len(split.val_pockets),
        "split_by": config.split_by,
        "selection_metric": config.selection_metric,
        "unsupported_metal_policy": config.unsupported_metal_policy,
        "train_metal_label_coverage": train_metal_coverage,
        "train_ec_label_coverage": train_ec_coverage,
        "val_metal_label_coverage": val_metal_coverage,
        "val_ec_label_coverage": val_ec_coverage,
        "train_feature_coverage": train_feature_coverage,
        "val_feature_coverage": val_feature_coverage,
        "split_leakage_check": leakage_check,
        "feature_load_report_excerpt": {
            "total_structure_files": feature_load_report.get("total_structure_files"),
            "loaded_structure_files": feature_load_report.get("loaded_structure_files"),
            "skipped_pockets": feature_load_report.get("skipped_pockets", [])[:10],
            "feature_fallbacks": feature_load_report.get("feature_fallbacks", [])[:10],
        }
        if feature_load_report is not None
        else None,
        "warnings": warnings,
    }


def run_preflight_checks(
    split: PocketSplit,
    config: TrainConfig,
    *,
    feature_load_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not split.train_pockets:
        raise ValueError("Preflight failed: training split is empty.")
    if config.val_fraction > 0.0 and not split.val_pockets:
        raise ValueError("Preflight failed: validation split is empty even though validation was requested.")
    if config.val_fraction == 0.0 and split.val_pockets:
        raise ValueError("Preflight failed: validation pockets exist even though --val-fraction is 0.")

    empty_train = find_pockets_without_residues(split.train_pockets)
    if empty_train:
        raise ValueError(f"Preflight failed: train pockets without residues: {empty_train[:5]}")

    empty_val = find_pockets_without_residues(split.val_pockets)
    if empty_val:
        raise ValueError(f"Preflight failed: validation pockets without residues: {empty_val[:5]}")

    report = build_preflight_report(
        split,
        config,
        feature_load_report=feature_load_report,
    )
    if len(report["train_metal_label_coverage"]["present_labels"]) < 2:
        raise ValueError("Preflight failed: train split covers fewer than 2 metal classes.")
    if len(report["train_ec_label_coverage"]["present_labels"]) < 2:
        raise ValueError("Preflight failed: train split covers fewer than 2 EC classes.")
    if report["split_leakage_check"]["overlap_count"] > 0:
        raise ValueError(
            "Preflight failed: train/validation leakage detected for "
            f"--split-by {config.split_by!r}: {report['split_leakage_check']['overlap_examples']}"
        )
    train_graph_probe = probe_graph_construction(split.train_pockets, config)
    val_graph_probe = probe_graph_construction(split.val_pockets, config)
    report["graph_probe"] = train_graph_probe
    report["train_graph_probe"] = train_graph_probe
    report["val_graph_probe"] = val_graph_probe
    return report


__all__ = [
    "build_preflight_report",
    "build_split_leakage_report",
    "find_pockets_without_residues",
    "label_coverage_summary",
    "probe_graph_construction",
    "run_preflight_checks",
]
