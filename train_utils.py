from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from data_structures import (
    DEFAULT_EDGE_RADIUS,
    EDGE_SOURCE_TO_INDEX,
    NORMALIZABLE_FEATURE_NAMES,
    PocketRecord,
)
from graph_construction import (
    pocket_to_pyg_data,
)
from label_schemes import (
    EC_TOP_LEVEL_LABELS,
    METAL_TARGET_LABELS,
    N_EC_CLASSES,
    N_METAL_CLASSES,
)
from model import GVPPocketClassifier
from training_data import (
    DEFAULT_STRUCTURE_DIR,
    load_smoke_test_pockets_from_dir,
)


@dataclass
class FeatureNormalizationStats:
    means: Dict[str, Tensor]
    stds: Dict[str, Tensor]
    clamp_value: float = 5.0


def compute_feature_normalization_stats(data_list: List[Data], clamp_value: float = 5.0) -> FeatureNormalizationStats:
    means: Dict[str, Tensor] = {}
    stds: Dict[str, Tensor] = {}

    for feature_name in NORMALIZABLE_FEATURE_NAMES:
        tensors = [getattr(data, feature_name) for data in data_list if hasattr(data, feature_name)]
        if not tensors:
            continue
        merged = torch.cat([tensor.float() for tensor in tensors], dim=0)
        mean = merged.mean(dim=0, keepdim=True)
        std = merged.std(dim=0, unbiased=False, keepdim=True)
        std = torch.where(std < 1e-6, torch.ones_like(std), std)
        means[feature_name] = mean
        stds[feature_name] = std

    return FeatureNormalizationStats(means=means, stds=stds, clamp_value=clamp_value)


def apply_feature_normalization(data: Data, stats: Optional[FeatureNormalizationStats]) -> Data:
    if stats is None:
        return data

    for feature_name, mean in stats.means.items():
        if not hasattr(data, feature_name):
            continue
        value = getattr(data, feature_name).float()
        std = stats.stds[feature_name].to(value.device)
        mean = mean.to(value.device)
        normalized = (value - mean) / std
        normalized = normalized.clamp(-stats.clamp_value, stats.clamp_value)
        setattr(data, feature_name, normalized)
    return data


def summarize_graph_dataset(
    pockets: List[PocketRecord],
    esm_dim: int,
    edge_radius: float = DEFAULT_EDGE_RADIUS,
    require_ring_edges: bool = False,
) -> List[Dict[str, Any]]:
    report: List[Dict[str, Any]] = []
    ring_idx = EDGE_SOURCE_TO_INDEX["ring"]
    # TODO- HERE I need to add a real-data report, not only this structural summary:
    # train/validation/test split description, leakage checks across homologs / UniProt / structure family,
    # and dataset-level counts for pockets with usable ring edges.

    for pocket in pockets:
        data = pocket_to_pyg_data(
            pocket,
            esm_dim=esm_dim,
            edge_radius=edge_radius,
            require_ring_edges=require_ring_edges,
        )
        edge_pairs = list(zip(data.edge_index[0].tolist(), data.edge_index[1].tolist()))
        duplicate_pairs = len(edge_pairs) - len(set(edge_pairs))
        ring_mask = data.edge_source_type[:, ring_idx] > 0.5
        report.append(
            {
                "pocket_id": pocket.pocket_id,
                "metal_count": int(pocket.metal_count()),
                "is_multinuclear": bool(pocket.is_multinuclear()),
                "n_residues": int(data.pos.size(0)),
                "n_edges": int(data.edge_index.size(1)),
                "n_radius_edges": int((~ring_mask).sum().item()),
                "n_ring_edges": int(ring_mask.sum().item()),
                "n_duplicate_pairs": int(duplicate_pairs),
            }
        )
    return report


class PocketGraphDataset(Dataset):
    def __init__(
        self,
        pockets: List[PocketRecord],
        esm_dim: int,
        edge_radius: float = DEFAULT_EDGE_RADIUS,
        normalization_stats: Optional[FeatureNormalizationStats] = None,
        require_ring_edges: bool = False,
    ):
        self.pockets = pockets
        self.esm_dim = esm_dim
        self.edge_radius = edge_radius
        self.normalization_stats = normalization_stats
        self.require_ring_edges = require_ring_edges

    @classmethod
    def fit_normalization_stats(
        cls,
        pockets: List[PocketRecord],
        esm_dim: int,
        edge_radius: float = DEFAULT_EDGE_RADIUS,
        clamp_value: float = 5.0,
        require_ring_edges: bool = False,
    ) -> FeatureNormalizationStats:
        data_list = [
            pocket_to_pyg_data(
                pocket,
                esm_dim=esm_dim,
                edge_radius=edge_radius,
                require_ring_edges=require_ring_edges,
            )
            for pocket in pockets
        ]
        return compute_feature_normalization_stats(data_list, clamp_value=clamp_value)

    def __len__(self) -> int:
        return len(self.pockets)

    def __getitem__(self, idx: int) -> Data:
        data = pocket_to_pyg_data(
            self.pockets[idx],
            esm_dim=self.esm_dim,
            edge_radius=self.edge_radius,
            require_ring_edges=self.require_ring_edges,
        )
        return apply_feature_normalization(data, self.normalization_stats)


def train_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, device: str = "cpu") -> float:
    model.train()
    total = 0.0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        model_outputs = model(batch)
        loss = model_outputs["loss"]
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += float(loss.item())

    return total / max(1, len(loader))


@torch.no_grad()
def evaluate_epoch(model: nn.Module, loader: DataLoader, device: str = "cpu") -> float:
    model.eval()
    total = 0.0

    for batch in loader:
        batch = batch.to(device)
        model_outputs = model(batch)
        loss = model_outputs["loss"]
        total += float(loss.item())

    return total / max(1, len(loader))


def balanced_class_weights_from_labels(labels: List[int], n_classes: int) -> Tensor:
    counts = torch.bincount(torch.tensor(labels, dtype=torch.long), minlength=n_classes).float()
    counts = torch.where(counts > 0, counts, torch.ones_like(counts))
    weights = counts.sum() / (counts * float(n_classes))
    return weights / weights.mean()


def balanced_class_weights_from_pockets(
    pockets: List[PocketRecord],
    n_metal_classes: int,
    n_ec_classes: int,
) -> Tuple[Tensor, Tensor]:
    # TODO- HERE I need to add the real class-distribution report for both heads
    # and confirm these class weights from the actual training split.
    metal_labels = [int(pocket.y_metal) for pocket in pockets if pocket.y_metal is not None]
    ec_labels = [int(pocket.y_ec) for pocket in pockets if pocket.y_ec is not None]
    metal_weights = balanced_class_weights_from_labels(metal_labels, n_metal_classes)
    ec_weights = balanced_class_weights_from_labels(ec_labels, n_ec_classes)
    return metal_weights, ec_weights


@torch.no_grad()
def predict_batch(model: nn.Module, loader: DataLoader, device: str = "cpu") -> Dict[str, Tensor]:
    model.eval()

    metal_logits_all = []
    ec_logits_all = []
    metal_y_all = []
    ec_y_all = []

    for batch in loader:
        batch = batch.to(device)
        model_outputs = model(batch)

        metal_logits_all.append(model_outputs["logits_metal"].cpu())
        ec_logits_all.append(model_outputs["logits_ec"].cpu())

        if hasattr(batch, "y_metal"):
            metal_y_all.append(batch.y_metal.cpu())
        if hasattr(batch, "y_ec"):
            ec_y_all.append(batch.y_ec.cpu())

    result = {
        "metal_logits": torch.cat(metal_logits_all, dim=0),
        "ec_logits": torch.cat(ec_logits_all, dim=0),
    }
    if metal_y_all:
        result["metal_y"] = torch.cat(metal_y_all, dim=0)
    if ec_y_all:
        result["ec_y"] = torch.cat(ec_y_all, dim=0)
    return result


@torch.no_grad()
def accuracy_from_logits(logits: Tensor, y: Tensor) -> float:
    # TODO- HERE I need to add the final evaluation metric choice for real experiments:
    # accuracy vs macro-F1 vs balanced accuracy vs per-class recall.
    pred = logits.argmax(dim=-1)
    return float((pred == y).float().mean().item())


def run_smoke_test(
    structure_dir: str | Path = DEFAULT_STRUCTURE_DIR,
    device: str = "cpu",
    esm_dim: int = 256,
    edge_radius: float = 10.0,
    max_cases: int = 4,
    batch_size: int = 2,
) -> None:
    structure_dir = Path(structure_dir)
    pockets = load_smoke_test_pockets_from_dir(
        structure_dir=structure_dir,
        max_cases=max_cases,
        require_full_labels=True,
    )

    graph_summary = summarize_graph_dataset(pockets, esm_dim=esm_dim, edge_radius=edge_radius)
    normalization_stats = PocketGraphDataset.fit_normalization_stats(
        pockets,
        esm_dim=esm_dim,
        edge_radius=edge_radius,
        clamp_value=5.0,
    )

    dataset = PocketGraphDataset(
        pockets,
        esm_dim=esm_dim,
        edge_radius=edge_radius,
        normalization_stats=normalization_stats,
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
