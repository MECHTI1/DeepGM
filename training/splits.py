from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from data_structures import PocketRecord
from label_schemes import EC_TOP_LEVEL_LABELS, METAL_TARGET_LABELS
from training.config import TrainConfig, VALID_SPLIT_BY_CHOICES
from training.feature_sources import build_pocket_feature_coverage
from training.labels import parse_structure_identity


@dataclass(frozen=True)
class PocketSplit:
    train_pockets: list[PocketRecord]
    val_pockets: list[PocketRecord]


def validate_split_by(split_by: str) -> str:
    if split_by not in VALID_SPLIT_BY_CHOICES:
        raise ValueError(
            f"Unsupported --split-by value: {split_by!r}. "
            f"Expected one of: {', '.join(repr(choice) for choice in VALID_SPLIT_BY_CHOICES)}."
        )
    return split_by


def pocket_split_key(pocket: PocketRecord, split_by: str) -> str:
    pdbid, chain, _ec = parse_structure_identity(pocket.structure_id)
    if split_by == "structure_id":
        return pocket.structure_id
    if split_by == "pdbid":
        return pdbid
    if split_by == "pdbid_chain":
        return f"{pdbid}__chain_{chain}"
    if split_by == "pocket_id":
        return pocket.pocket_id
    raise AssertionError(f"Unhandled split_by value: {split_by!r}")


def split_pockets(
    pockets: list[PocketRecord],
    val_fraction: float,
    split_by: str,
    seed: int,
    task: str = "joint",
) -> PocketSplit:
    split_by = validate_split_by(split_by)
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError(f"--val-fraction must be in [0, 1), got {val_fraction}")
    if val_fraction == 0.0:
        return PocketSplit(train_pockets=pockets, val_pockets=[])

    grouped: dict[str, list[PocketRecord]] = {}
    for pocket in pockets:
        grouped.setdefault(pocket_split_key(pocket, split_by), []).append(pocket)

    group_items = list(grouped.items())
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(group_items), generator=generator).tolist()
    shuffled = [group_items[idx] for idx in order]
    shuffled.sort(key=lambda item: len(item[1]), reverse=True)

    target_val_size = max(1, int(round(len(pockets) * val_fraction)))
    val_pockets: list[PocketRecord] = []
    train_pockets: list[PocketRecord] = []
    val_count = 0
    remaining_pocket_count = len(pockets)
    desired_val_label_counts = desired_label_counts_for_split(pockets, task=task, val_fraction=val_fraction)
    current_val_label_counts: dict[str, int] = {}

    for _group_key, group_pockets in shuffled:
        group_size = len(group_pockets)
        remaining_pocket_count -= group_size
        must_assign_to_val = val_count < target_val_size and (val_count + remaining_pocket_count) < target_val_size
        proposed_val_label_counts = merge_label_counts(
            current_val_label_counts,
            label_counts_for_pockets(group_pockets, task=task),
        )
        current_penalty = val_assignment_penalty(
            val_count=val_count,
            target_val_size=target_val_size,
            current_label_counts=current_val_label_counts,
            desired_label_counts=desired_val_label_counts,
        )
        proposed_penalty = val_assignment_penalty(
            val_count=val_count + group_size,
            target_val_size=target_val_size,
            current_label_counts=proposed_val_label_counts,
            desired_label_counts=desired_val_label_counts,
        )

        if must_assign_to_val or (val_count < target_val_size and proposed_penalty <= current_penalty):
            val_pockets.extend(group_pockets)
            val_count += group_size
            current_val_label_counts = proposed_val_label_counts
        else:
            train_pockets.extend(group_pockets)

    if not train_pockets or not val_pockets:
        raise ValueError(
            "Validation split produced an empty train or validation set. "
            "Adjust --val-fraction or --split-by."
        )

    return PocketSplit(train_pockets=train_pockets, val_pockets=val_pockets)


def task_label_keys_for_pocket(pocket: PocketRecord, task: str) -> list[str]:
    keys: list[str] = []
    if task in ("joint", "metal") and pocket.y_metal is not None:
        keys.append(f"metal:{int(pocket.y_metal)}")
    if task in ("joint", "ec") and pocket.y_ec is not None:
        keys.append(f"ec:{int(pocket.y_ec)}")
    return keys


def label_counts_for_pockets(pockets: list[PocketRecord], task: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for pocket in pockets:
        for key in task_label_keys_for_pocket(pocket, task):
            counts[key] = counts.get(key, 0) + 1
    return counts


def merge_label_counts(base: dict[str, int], extra: dict[str, int]) -> dict[str, int]:
    merged = dict(base)
    for key, value in extra.items():
        merged[key] = merged.get(key, 0) + value
    return merged


def desired_label_counts_for_split(
    pockets: list[PocketRecord],
    *,
    task: str,
    val_fraction: float,
) -> dict[str, float]:
    return {
        key: count * val_fraction
        for key, count in label_counts_for_pockets(pockets, task).items()
    }


def val_assignment_penalty(
    *,
    val_count: int,
    target_val_size: int,
    current_label_counts: dict[str, int],
    desired_label_counts: dict[str, float],
) -> float:
    size_penalty = abs(val_count - target_val_size)
    label_penalty = sum(
        abs(current_label_counts.get(key, 0) - desired_count)
        for key, desired_count in desired_label_counts.items()
    )
    overshoot_penalty = max(0, val_count - target_val_size)
    return (1000.0 * overshoot_penalty) + (2.0 * size_penalty) + (10.0 * label_penalty)


def count_labels(
    pockets: list[PocketRecord],
    attr_name: str,
    label_map: dict[int, str],
) -> dict[str, int]:
    counts = {label_name: 0 for label_name in label_map.values()}
    for pocket in pockets:
        label_idx = getattr(pocket, attr_name)
        if label_idx is None:
            continue
        counts[label_map[int(label_idx)]] += 1
    return counts


def build_dataset_summary(
    split: PocketSplit,
    config: TrainConfig,
    feature_load_report: dict[str, Any],
) -> dict[str, Any]:
    return {
        "structure_dir": str(config.structure_dir),
        "summary_csv": str(config.summary_csv),
        "esm_embeddings_dir": config.esm_embeddings_dir,
        "external_features_root_dir": config.external_features_root_dir,
        "external_feature_source": config.external_feature_source,
        "n_train_pockets": len(split.train_pockets),
        "n_val_pockets": len(split.val_pockets),
        "task": config.task,
        "val_fraction": config.val_fraction,
        "split_by": config.split_by,
        "selection_metric": config.selection_metric,
        "unsupported_metal_policy": config.unsupported_metal_policy,
        "invalid_structure_policy": config.invalid_structure_policy,
        "feature_load_report": feature_load_report,
        "train_feature_coverage": build_pocket_feature_coverage(split.train_pockets),
        "val_feature_coverage": build_pocket_feature_coverage(split.val_pockets),
        "train_metal_distribution": count_labels(split.train_pockets, "y_metal", METAL_TARGET_LABELS),
        "train_ec_distribution": count_labels(split.train_pockets, "y_ec", EC_TOP_LEVEL_LABELS),
        "val_metal_distribution": count_labels(split.val_pockets, "y_metal", METAL_TARGET_LABELS),
        "val_ec_distribution": count_labels(split.val_pockets, "y_ec", EC_TOP_LEVEL_LABELS),
    }
