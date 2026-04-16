from __future__ import annotations

"""Smoke-test runner for the training pipeline."""

from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

from data_structures import DEFAULT_EDGE_RADIUS
from label_schemes import EC_TOP_LEVEL_LABELS, METAL_TARGET_LABELS, N_EC_CLASSES, N_METAL_CLASSES
from model import GVPPocketClassifier
from training.data import load_smoke_test_pockets_from_dir
from training.defaults import DEFAULT_STRUCTURE_DIR
from training.esm_feature_loading import DEFAULT_ESMC_EMBED_DIM
from training.graph_dataset import (
    FeatureNormalizationStats,
    PocketGraphDataset,
    build_graph_data_list,
    compute_feature_normalization_stats,
    summarize_graph_dataset,
)
from training.loop import (
    accuracy_from_logits,
    balanced_class_weights_from_pockets,
    evaluate_epoch,
    predict_batch,
    train_epoch,
)


def run_smoke_test(
    structure_dir: str | Path = DEFAULT_STRUCTURE_DIR,
    device: str = "cpu",
    esm_dim: int = DEFAULT_ESMC_EMBED_DIM,
    edge_radius: float = DEFAULT_EDGE_RADIUS,
    max_cases: int = 4,
    batch_size: int = 2,
) -> None:
    structure_dir = Path(structure_dir)
    pockets = load_smoke_test_pockets_from_dir(
        structure_dir=structure_dir,
        max_cases=max_cases,
        require_full_labels=True,
        esm_dim=esm_dim,
    )

    graph_summary = summarize_graph_dataset(pockets, esm_dim=esm_dim, edge_radius=edge_radius)
    graph_data_list = build_graph_data_list(
        pockets,
        esm_dim=esm_dim,
        edge_radius=edge_radius,
    )
    normalization_stats = compute_feature_normalization_stats(graph_data_list, clamp_value=5.0)

    dataset = PocketGraphDataset(
        pockets,
        esm_dim=esm_dim,
        edge_radius=edge_radius,
        normalization_stats=normalization_stats,
        precomputed_data=graph_data_list,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_have_supervision = all(pocket.y_metal is not None and pocket.y_ec is not None for pocket in pockets)
    metal_class_weights = None
    ec_class_weights = None
    if all_have_supervision:
        metal_class_weights, ec_class_weights = balanced_class_weights_from_pockets(
            pockets,
            n_metal_classes=N_METAL_CLASSES,
            n_ec_classes=N_EC_CLASSES,
        )

    model = GVPPocketClassifier(
        esm_dim=esm_dim,
        metal_class_weights=metal_class_weights,
        ec_class_weights=ec_class_weights,
    ).to(device)

    print(f"Smoke-test structure dir: {structure_dir}")
    print(f"EC top-level labels: {EC_TOP_LEVEL_LABELS}")
    print(f"Metal target labels: {METAL_TARGET_LABELS}")
    print("Metal targets are inferred per pocket from parsed structure metal symbols.")
    print("Smoke-test pockets:", [pocket.pocket_id for pocket in pockets])

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    loss = train_epoch(model, loader, optimizer, device=device)
    print(f"Smoke-test train loss: {loss:.4f}")

    result = predict_batch(model, loader, device=device)
    print("Graph summary sample:", graph_summary[0])
    print("Metal logits shape:", tuple(result["metal_logits"].shape))
    print("EC logits shape:", tuple(result["ec_logits"].shape))
    if all_have_supervision and "metal_y" in result:
        print("Metal acc:", accuracy_from_logits(result["metal_logits"], result["metal_y"]))
    if all_have_supervision and "ec_y" in result:
        print("EC acc:", accuracy_from_logits(result["ec_logits"], result["ec_y"]))
    print("Smoke test completed successfully.")
