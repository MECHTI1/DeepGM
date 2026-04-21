from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from torch_geometric.loader import DataLoader

from evaluate_legacy_test_set import aggregate_structure_logits, load_checkpoint_model
from label_schemes import EC_TOP_LEVEL_LABELS, METAL_TARGET_LABELS
from training.data import load_labeled_pockets_with_report_from_dir
from training.graph_dataset import PocketGraphDataset, build_graph_data_list
from training.loop import classification_metrics_from_logits, evaluate_epoch_with_predictions


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a trained DeepGM checkpoint on a labeled structure directory using the same "
            "data-loading path as training/validation."
        )
    )
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--structure-dir", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, default=None)
    parser.add_argument("--esm-embeddings-dir", type=Path, required=True)
    parser.add_argument("--external-features-root-dir", type=Path, required=True)
    parser.add_argument(
        "--external-feature-source",
        type=str,
        default="updated",
        choices=("auto", "updated", "bluues_rosetta"),
    )
    parser.add_argument(
        "--unsupported-metal-policy",
        type=str,
        default="error",
        choices=("error", "skip"),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser


def metrics_payload_from_logits(logits: torch.Tensor, y: torch.Tensor) -> dict[str, Any]:
    metrics = classification_metrics_from_logits(logits, y)
    return {
        "accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "macro_f1": metrics["macro_f1"],
        "per_class_recall": metrics["per_class_recall"],
    }


def build_result_payload(
    *,
    checkpoint_path: Path,
    load_result,
    graph_error_sample: list[dict[str, str]],
    pocket_metal_logits: torch.Tensor,
    pocket_ec_logits: torch.Tensor,
    structure_ids: list[str],
    structure_metal_logits: torch.Tensor,
    structure_ec_logits: torch.Tensor,
    structure_y: torch.Tensor,
) -> dict[str, Any]:
    pocket_y_metal = torch.tensor([int(pocket.y_metal) for pocket in load_result.pockets], dtype=torch.long)
    pocket_y_ec = torch.tensor([int(pocket.y_ec) for pocket in load_result.pockets], dtype=torch.long)

    pocket_metal_metrics = metrics_payload_from_logits(pocket_metal_logits, pocket_y_metal)
    pocket_ec_metrics = metrics_payload_from_logits(pocket_ec_logits, pocket_y_ec)
    structure_metal_metrics = metrics_payload_from_logits(structure_metal_logits, structure_y)
    predicted_structure_ec = structure_ec_logits.argmax(dim=-1).tolist()
    ec_prediction_distribution = Counter(predicted_structure_ec)

    per_structure = []
    for structure_id, predicted_metal, predicted_ec, true_metal in zip(
        structure_ids,
        structure_metal_logits.argmax(dim=-1).tolist(),
        predicted_structure_ec,
        structure_y.tolist(),
    ):
        per_structure.append(
            {
                "structure_id": structure_id,
                "true_metal_label_id": int(true_metal),
                "true_metal_label_name": METAL_TARGET_LABELS[int(true_metal)],
                "predicted_metal_label_id": int(predicted_metal),
                "predicted_metal_label_name": METAL_TARGET_LABELS[int(predicted_metal)],
                "predicted_ec_label_id": int(predicted_ec),
                "predicted_ec_label_name": EC_TOP_LEVEL_LABELS[int(predicted_ec)],
            }
        )

    return {
        "checkpoint_path": str(checkpoint_path),
        "n_loaded_pockets": len(load_result.pockets),
        "n_loaded_structures": len({p.structure_id for p in load_result.pockets}),
        "feature_load_report": load_result.feature_report,
        "n_graph_errors": len(graph_error_sample),
        "graph_error_sample": graph_error_sample[:25],
        "pocket_metal_metrics": pocket_metal_metrics,
        "pocket_ec_metrics": pocket_ec_metrics,
        "pocket_joint_balanced_acc": float(
            (pocket_metal_metrics["balanced_accuracy"] + pocket_ec_metrics["balanced_accuracy"]) / 2.0
        ),
        "structure_metal_metrics": structure_metal_metrics,
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
    esm_dim = int(checkpoint["config"]["esm_dim"])
    edge_radius = float(checkpoint["config"]["edge_radius"])
    require_ring_edges = bool(checkpoint["config"].get("require_ring_edges", False))
    require_esm_embeddings = bool(checkpoint["config"].get("require_esm_embeddings", True))
    require_external_features = bool(checkpoint["config"].get("require_external_features", True))

    load_result = load_labeled_pockets_with_report_from_dir(
        structure_dir=args.structure_dir,
        require_full_labels=True,
        summary_csv=args.summary_csv,
        esm_dim=esm_dim,
        esm_embeddings_dir=args.esm_embeddings_dir,
        require_esm_embeddings=require_esm_embeddings,
        external_features_root_dir=args.external_features_root_dir,
        external_feature_source=args.external_feature_source,
        require_external_features=require_external_features,
        unsupported_metal_policy=args.unsupported_metal_policy,
    )
    if not load_result.pockets:
        raise ValueError("No labeled pockets were loaded from the requested structure directory.")

    graph_pockets = []
    graph_data = []
    graph_error_sample: list[dict[str, str]] = []
    for pocket in load_result.pockets:
        try:
            pocket_graph_data = build_graph_data_list(
                [pocket],
                esm_dim=esm_dim,
                edge_radius=edge_radius,
                require_ring_edges=require_ring_edges,
            )[0]
        except Exception as exc:
            graph_error_sample.append(
                {
                    "structure_id": pocket.structure_id,
                    "pocket_id": pocket.pocket_id,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            continue
        graph_pockets.append(pocket)
        graph_data.append(pocket_graph_data)
    if not graph_pockets:
        raise ValueError("No valid graph pockets remained after graph construction.")

    load_result = type(load_result)(
        pockets=graph_pockets,
        feature_report=load_result.feature_report,
    )
    loader = DataLoader(
        PocketGraphDataset(
            load_result.pockets,
            esm_dim=esm_dim,
            edge_radius=edge_radius,
            normalization_stats=normalization_stats,
            require_ring_edges=require_ring_edges,
            precomputed_data=graph_data,
        ),
        batch_size=args.batch_size,
        shuffle=False,
    )
    predictions = evaluate_epoch_with_predictions(model, loader, device=args.device)
    structure_ids, structure_metal_logits, structure_ec_logits, structure_y, _pocket_counts = aggregate_structure_logits(
        pockets=load_result.pockets,
        metal_logits=predictions["metal_logits"],
        ec_logits=predictions["ec_logits"],
    )
    payload = build_result_payload(
        checkpoint_path=args.checkpoint_path,
        load_result=load_result,
        graph_error_sample=graph_error_sample,
        pocket_metal_logits=predictions["metal_logits"],
        pocket_ec_logits=predictions["ec_logits"],
        structure_ids=structure_ids,
        structure_metal_logits=structure_metal_logits,
        structure_ec_logits=structure_ec_logits,
        structure_y=structure_y,
    )

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Loaded structures: {payload['n_loaded_structures']}")
    print(f"Loaded pockets: {payload['n_loaded_pockets']}")
    print(f"Pocket metal balanced accuracy: {payload['pocket_metal_metrics']['balanced_accuracy']:.4f}")
    print(f"Pocket EC balanced accuracy: {payload['pocket_ec_metrics']['balanced_accuracy']:.4f}")
    print(f"Pocket joint balanced accuracy: {payload['pocket_joint_balanced_acc']:.4f}")
    print(f"Structure metal balanced accuracy: {payload['structure_metal_metrics']['balanced_accuracy']:.4f}")
    if args.output_json is not None:
        print(f"Wrote labeled-structure evaluation report to {args.output_json}")


if __name__ == "__main__":
    main()
