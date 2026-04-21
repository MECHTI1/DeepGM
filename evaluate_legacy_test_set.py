from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch_geometric.loader import DataLoader

from label_schemes import EC_TOP_LEVEL_LABELS, METAL_TARGET_LABELS, N_EC_CLASSES, N_METAL_CLASSES
from model import GVPPocketClassifier
from training.feature_sources import attach_structure_features_to_pocket, load_structure_feature_sources
from training.graph_dataset import FeatureNormalizationStats, PocketGraphDataset, build_graph_data_list
from training.loop import classification_metrics_from_logits
from graph.construction import extract_metal_pockets_from_structure, parse_structure_file


LEGACY_TEST_CSV = Path("prepare_training_and_test_set/pinmymetal_files/classmodel_test_set")
LEGACY_TEST_STRUCTURE_ROOT = Path("/media/Data/pinmymetal_sets/test")
LEGACY_LABEL_TO_TARGET = {
    "1": 0,  # Mn
    "6": 1,  # Cu
    "7": 2,  # Zn
    "2": 3,  # Class VIII (Fe / Co / Ni group)
}


@dataclass(frozen=True)
class LegacyTestLabelLoadResult:
    label_by_pdbid: dict[str, int]
    mixed_pdbids: dict[str, list[str]]
    unsupported_label_counts: dict[str, int]


@dataclass(frozen=True)
class LegacyTestDataset:
    pockets: list[Any]
    graph_data: list[Any]
    missing_structures: list[str]
    structures_without_pockets: list[str]
    structure_errors: list[dict[str, str]]
    graph_errors: list[dict[str, str]]
    feature_fallbacks: list[dict[str, str]]
    pockets_per_structure: dict[str, int]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a trained DeepGM checkpoint on the legacy raw test set. "
            "This path reports metal-class metrics only because the legacy test CSV does not carry the "
            "EC labels required for the current joint loss."
        )
    )
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--test-csv", type=Path, default=LEGACY_TEST_CSV)
    parser.add_argument("--structure-root", type=Path, default=LEGACY_TEST_STRUCTURE_ROOT)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--esm-embeddings-dir",
        type=Path,
        default=None,
        help=(
            "Optional raw-test embeddings directory. If omitted, the evaluator zero-fills missing "
            "ESM embeddings instead of trying to reuse train-set embeddings with mismatched names."
        ),
    )
    parser.add_argument(
        "--external-features-root-dir",
        type=Path,
        default=None,
        help=(
            "Optional raw-test external feature root. If omitted, the evaluator zero-fills missing "
            "external features."
        ),
    )
    parser.add_argument(
        "--external-feature-source",
        type=str,
        default="updated",
        choices=("auto", "updated", "bluues_rosetta"),
        help=(
            "How to resolve external feature directories. Use 'updated' for residue_features.json roots "
            "and 'bluues_rosetta' for legacy per-structure feature folders."
        ),
    )
    parser.add_argument("--output-json", type=Path, default=None)
    return parser


def _tensor_dict_from_payload(payload: dict[str, Any]) -> dict[str, torch.Tensor]:
    converted: dict[str, torch.Tensor] = {}
    for key, value in payload.items():
        converted[key] = value if isinstance(value, torch.Tensor) else torch.tensor(value)
    return converted


def normalization_stats_from_checkpoint(payload: dict[str, Any]) -> FeatureNormalizationStats:
    return FeatureNormalizationStats(
        means=_tensor_dict_from_payload(dict(payload["means"])),
        stds=_tensor_dict_from_payload(dict(payload["stds"])),
        clamp_value=float(payload.get("clamp_value", 5.0)),
    )


def load_checkpoint_model(
    checkpoint_path: Path,
    *,
    device: str,
) -> tuple[dict[str, Any], GVPPocketClassifier, FeatureNormalizationStats]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = dict(checkpoint["config"])

    model = GVPPocketClassifier(
        esm_dim=int(config["esm_dim"]),
        n_metal=len(checkpoint.get("metal_labels", METAL_TARGET_LABELS)),
        n_ec=len(checkpoint.get("ec_labels", {idx: str(idx) for idx in range(N_EC_CLASSES)})),
        metal_class_weights=torch.ones(N_METAL_CLASSES, dtype=torch.float32),
        ec_class_weights=torch.ones(N_EC_CLASSES, dtype=torch.float32),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    normalization_stats = normalization_stats_from_checkpoint(checkpoint["normalization_stats"])
    return checkpoint, model, normalization_stats


def load_legacy_test_labels(test_csv: Path) -> LegacyTestLabelLoadResult:
    labels_by_pdbid: dict[str, set[str]] = {}
    unsupported_counts: Counter[str] = Counter()

    with test_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            pdbid = row["pdbid"].strip().lower()
            legacy_label = row["label_metal"].strip()
            if legacy_label not in LEGACY_LABEL_TO_TARGET:
                unsupported_counts[legacy_label] += 1
                continue
            labels_by_pdbid.setdefault(pdbid, set()).add(legacy_label)

    label_by_pdbid: dict[str, int] = {}
    mixed_pdbids: dict[str, list[str]] = {}
    for pdbid, labels in labels_by_pdbid.items():
        if len(labels) != 1:
            mixed_pdbids[pdbid] = sorted(labels)
            continue
        only_label = next(iter(labels))
        label_by_pdbid[pdbid] = LEGACY_LABEL_TO_TARGET[only_label]

    return LegacyTestLabelLoadResult(
        label_by_pdbid=label_by_pdbid,
        mixed_pdbids=mixed_pdbids,
        unsupported_label_counts=dict(sorted(unsupported_counts.items())),
    )


def resolve_legacy_structure_path(structure_root: Path, pdbid: str) -> Path | None:
    candidates = (
        structure_root / f"{pdbid}.cif",
        structure_root / f"{pdbid}.pdb",
        structure_root / "cif" / f"{pdbid}.cif",
        structure_root / "pdb" / f"{pdbid}.pdb",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def build_legacy_test_dataset(
    *,
    label_by_pdbid: dict[str, int],
    structure_root: Path,
    esm_dim: int,
    edge_radius: float,
    require_ring_edges: bool,
    embeddings_dir: Path | None,
    external_features_root_dir: Path | None,
    external_feature_source: str,
) -> LegacyTestDataset:
    pockets: list[Any] = []
    graph_data_list: list[Any] = []
    missing_structures: list[str] = []
    structures_without_pockets: list[str] = []
    structure_errors: list[dict[str, str]] = []
    graph_errors: list[dict[str, str]] = []
    feature_fallbacks: list[dict[str, str]] = []
    pockets_per_structure: dict[str, int] = {}
    resolved_embeddings_dir = embeddings_dir if embeddings_dir is not None else structure_root / "__missing_embeddings__"

    for pdbid, label_id in sorted(label_by_pdbid.items()):
        structure_path = resolve_legacy_structure_path(structure_root, pdbid)
        if structure_path is None:
            missing_structures.append(pdbid)
            continue

        try:
            structure = parse_structure_file(str(structure_path), structure_id=pdbid)
            extracted_pockets = extract_metal_pockets_from_structure(structure, structure_id=pdbid)
            if not extracted_pockets:
                structures_without_pockets.append(pdbid)
                continue

            structure_feature_fallbacks: list[dict[str, str]] = []
            feature_sources = load_structure_feature_sources(
                structure=structure,
                structure_path=structure_path,
                structure_root=structure_root,
                embeddings_dir=resolved_embeddings_dir,
                require_esm_embeddings=False,
                feature_root_dir=external_features_root_dir or structure_root,
                external_feature_source=external_feature_source,
                require_external_features=False,
                feature_fallbacks=structure_feature_fallbacks,
            )

            kept_count = 0
            for pocket in extracted_pockets:
                pocket.metadata["source_path"] = str(structure_path)
                attach_structure_features_to_pocket(
                    pocket,
                    feature_sources=feature_sources,
                    esm_dim=esm_dim,
                    require_esm_embeddings=False,
                    require_external_features=False,
                    structure_path=structure_path,
                )
                pocket.y_metal = label_id
                try:
                    graph_data = build_graph_data_list(
                        [pocket],
                        esm_dim=esm_dim,
                        edge_radius=edge_radius,
                        require_ring_edges=require_ring_edges,
                    )
                except Exception as exc:
                    graph_errors.append(
                        {
                            "pdbid": pdbid,
                            "pocket_id": pocket.pocket_id,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                    )
                    continue
                pockets.append(pocket)
                graph_data_list.append(graph_data[0])
                kept_count += 1

            feature_fallbacks.extend(structure_feature_fallbacks)
            if kept_count == 0:
                structures_without_pockets.append(pdbid)
                continue
            pockets_per_structure[pdbid] = kept_count
        except Exception as exc:
            structure_errors.append(
                {
                    "pdbid": pdbid,
                    "structure_path": str(structure_path),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

    return LegacyTestDataset(
        pockets=pockets,
        graph_data=graph_data_list,
        missing_structures=missing_structures,
        structures_without_pockets=structures_without_pockets,
        structure_errors=structure_errors,
        graph_errors=graph_errors,
        feature_fallbacks=feature_fallbacks,
        pockets_per_structure=pockets_per_structure,
    )


@torch.no_grad()
def predict_model_logits(
    model: GVPPocketClassifier,
    loader: DataLoader,
    *,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    metal_logits_all: list[torch.Tensor] = []
    ec_logits_all: list[torch.Tensor] = []
    model.eval()
    for batch in loader:
        batch = batch.to(device)
        outputs = model(batch)
        metal_logits_all.append(outputs["logits_metal"].cpu())
        ec_logits_all.append(outputs["logits_ec"].cpu())
    if not metal_logits_all:
        return (
            torch.empty((0, N_METAL_CLASSES), dtype=torch.float32),
            torch.empty((0, N_EC_CLASSES), dtype=torch.float32),
        )
    return (
        torch.cat(metal_logits_all, dim=0),
        torch.cat(ec_logits_all, dim=0),
    )


def aggregate_structure_logits(
    *,
    pockets: list[Any],
    metal_logits: torch.Tensor,
    ec_logits: torch.Tensor,
) -> tuple[list[str], torch.Tensor, torch.Tensor, torch.Tensor, dict[str, int]]:
    structure_ids: list[str] = []
    true_labels: list[int] = []
    aggregated_metal_logits: list[torch.Tensor] = []
    aggregated_ec_logits: list[torch.Tensor] = []
    pocket_counts: dict[str, int] = {}

    by_structure: dict[str, list[int]] = {}
    for idx, pocket in enumerate(pockets):
        by_structure.setdefault(pocket.structure_id, []).append(idx)

    for structure_id, indices in sorted(by_structure.items()):
        structure_ids.append(structure_id)
        true_labels.append(int(pockets[indices[0]].y_metal))
        aggregated_metal_logits.append(metal_logits[indices].mean(dim=0))
        aggregated_ec_logits.append(ec_logits[indices].mean(dim=0))
        pocket_counts[structure_id] = len(indices)

    return (
        structure_ids,
        torch.stack(aggregated_metal_logits, dim=0),
        torch.stack(aggregated_ec_logits, dim=0),
        torch.tensor(true_labels, dtype=torch.long),
        pocket_counts,
    )


def metrics_payload_from_logits(logits: torch.Tensor, y: torch.Tensor) -> dict[str, Any]:
    metrics = classification_metrics_from_logits(logits, y)
    metrics["predicted_labels"] = logits.argmax(dim=-1).tolist()
    metrics["true_labels"] = y.tolist()
    return metrics


def build_result_payload(
    *,
    checkpoint_path: Path,
    label_result: LegacyTestLabelLoadResult,
    dataset: LegacyTestDataset,
    pocket_metal_logits: torch.Tensor,
    pocket_ec_logits: torch.Tensor,
    structure_ids: list[str],
    structure_metal_logits: torch.Tensor,
    structure_ec_logits: torch.Tensor,
    structure_y: torch.Tensor,
) -> dict[str, Any]:
    pocket_y = torch.tensor([int(pocket.y_metal) for pocket in dataset.pockets], dtype=torch.long)
    pocket_metrics = metrics_payload_from_logits(pocket_metal_logits, pocket_y)
    structure_metrics = metrics_payload_from_logits(structure_metal_logits, structure_y)
    predicted_structure_labels = structure_metal_logits.argmax(dim=-1).tolist()
    predicted_structure_ec = structure_ec_logits.argmax(dim=-1).tolist()
    predicted_pocket_ec = pocket_ec_logits.argmax(dim=-1).tolist()
    ec_prediction_distribution = Counter(predicted_structure_ec)

    per_structure = []
    for structure_id, predicted_label, predicted_ec, true_label in zip(
        structure_ids,
        predicted_structure_labels,
        predicted_structure_ec,
        structure_y.tolist(),
    ):
        per_structure.append(
            {
                "pdbid": structure_id,
                "true_label_id": int(true_label),
                "true_label_name": METAL_TARGET_LABELS[int(true_label)],
                "predicted_label_id": int(predicted_label),
                "predicted_label_name": METAL_TARGET_LABELS[int(predicted_label)],
                "predicted_ec_label_id": int(predicted_ec),
                "predicted_ec_label_name": EC_TOP_LEVEL_LABELS[int(predicted_ec)],
                "n_pockets": dataset.pockets_per_structure.get(structure_id, 0),
            }
        )

    return {
        "checkpoint_path": str(checkpoint_path),
        "n_labeled_structures_from_csv": len(label_result.label_by_pdbid),
        "n_mixed_label_structures_skipped": len(label_result.mixed_pdbids),
        "mixed_label_structures": label_result.mixed_pdbids,
        "unsupported_legacy_label_counts": label_result.unsupported_label_counts,
        "n_missing_structure_files": len(dataset.missing_structures),
        "missing_structure_files": dataset.missing_structures,
        "n_structures_without_pockets": len(dataset.structures_without_pockets),
        "structures_without_pockets": dataset.structures_without_pockets,
        "n_structure_errors": len(dataset.structure_errors),
        "structure_errors": dataset.structure_errors[:25],
        "n_graph_errors": len(dataset.graph_errors),
        "graph_error_sample": dataset.graph_errors[:25],
        "n_feature_fallbacks": len(dataset.feature_fallbacks),
        "feature_fallback_sample": dataset.feature_fallbacks[:25],
        "n_pocket_predictions": len(dataset.pockets),
        "n_structure_predictions": len(structure_ids),
        "pocket_metrics": pocket_metrics,
        "structure_metrics": structure_metrics,
        "pocket_predicted_ec_labels": predicted_pocket_ec,
        "structure_predicted_ec_distribution": {
            EC_TOP_LEVEL_LABELS[int(label_id)]: count
            for label_id, count in sorted(ec_prediction_distribution.items())
        },
        "per_structure_predictions": per_structure,
    }


def main() -> None:
    args = build_arg_parser().parse_args()
    checkpoint, model, normalization_stats = load_checkpoint_model(
        args.checkpoint_path,
        device=args.device,
    )
    label_result = load_legacy_test_labels(args.test_csv)
    dataset = build_legacy_test_dataset(
        label_by_pdbid=label_result.label_by_pdbid,
        structure_root=args.structure_root,
        esm_dim=int(checkpoint["config"]["esm_dim"]),
        edge_radius=float(checkpoint["config"]["edge_radius"]),
        require_ring_edges=bool(checkpoint["config"].get("require_ring_edges", False)),
        embeddings_dir=args.esm_embeddings_dir,
        external_features_root_dir=args.external_features_root_dir,
        external_feature_source=args.external_feature_source,
    )
    if not dataset.pockets:
        raise ValueError("Legacy test evaluator could not build any pockets.")

    loader = DataLoader(
        PocketGraphDataset(
            dataset.pockets,
            esm_dim=int(checkpoint["config"]["esm_dim"]),
            edge_radius=float(checkpoint["config"]["edge_radius"]),
            normalization_stats=normalization_stats,
            require_ring_edges=bool(checkpoint["config"].get("require_ring_edges", False)),
            precomputed_data=dataset.graph_data,
        ),
        batch_size=args.batch_size,
        shuffle=False,
    )
    pocket_metal_logits, pocket_ec_logits = predict_model_logits(model, loader, device=args.device)
    structure_ids, structure_metal_logits, structure_ec_logits, structure_y, _pocket_counts = aggregate_structure_logits(
        pockets=dataset.pockets,
        metal_logits=pocket_metal_logits,
        ec_logits=pocket_ec_logits,
    )
    payload = build_result_payload(
        checkpoint_path=args.checkpoint_path,
        label_result=label_result,
        dataset=dataset,
        pocket_metal_logits=pocket_metal_logits,
        pocket_ec_logits=pocket_ec_logits,
        structure_ids=structure_ids,
        structure_metal_logits=structure_metal_logits,
        structure_ec_logits=structure_ec_logits,
        structure_y=structure_y,
    )

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    structure_metrics = payload["structure_metrics"]
    print(f"Legacy test structures evaluated: {payload['n_structure_predictions']}")
    print(f"Legacy test pockets evaluated: {payload['n_pocket_predictions']}")
    print(f"Missing structure files: {payload['n_missing_structure_files']}")
    print(f"Structures without pockets: {payload['n_structures_without_pockets']}")
    print(f"Structure-level metal accuracy: {structure_metrics['accuracy']:.4f}")
    print(f"Structure-level metal balanced accuracy: {structure_metrics['balanced_accuracy']:.4f}")
    print(f"Structure-level metal macro F1: {structure_metrics['macro_f1']:.4f}")
    print(f"Structure-level predicted EC distribution: {payload['structure_predicted_ec_distribution']}")
    if args.output_json is not None:
        print(f"Wrote legacy test report to {args.output_json}")


if __name__ == "__main__":
    main()
