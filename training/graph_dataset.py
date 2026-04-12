from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import Dataset
from torch_geometric.data import Data

from data_structures import DEFAULT_EDGE_RADIUS, EDGE_SOURCE_TO_INDEX, NORMALIZABLE_FEATURE_NAMES, PocketRecord
from graph.construction import pocket_to_pyg_data


@dataclass
class FeatureNormalizationStats:
    means: dict[str, Tensor]
    stds: dict[str, Tensor]
    clamp_value: float = 5.0


def compute_feature_normalization_stats(
    data_list: list[Data],
    clamp_value: float = 5.0,
) -> FeatureNormalizationStats:
    means: dict[str, Tensor] = {}
    stds: dict[str, Tensor] = {}

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


def apply_feature_normalization(data: Data, stats: FeatureNormalizationStats | None) -> Data:
    if stats is None:
        return data

    for feature_name, mean in stats.means.items():
        if not hasattr(data, feature_name):
            continue
        value = getattr(data, feature_name).float()
        std = stats.stds[feature_name].to(value.device)
        normalized = (value - mean.to(value.device)) / std
        setattr(data, feature_name, normalized.clamp(-stats.clamp_value, stats.clamp_value))
    return data


def summarize_graph_dataset(
    pockets: list[PocketRecord],
    esm_dim: int,
    edge_radius: float = DEFAULT_EDGE_RADIUS,
    require_ring_edges: bool = False,
) -> list[dict[str, Any]]:
    report: list[dict[str, Any]] = []
    ring_idx = EDGE_SOURCE_TO_INDEX["ring"]

    for pocket in pockets:
        data = pocket_to_pyg_data(
            pocket,
            esm_dim=esm_dim,
            edge_radius=edge_radius,
            require_ring_edges=require_ring_edges,
        )
        edge_pairs = list(zip(data.edge_index[0].tolist(), data.edge_index[1].tolist()))
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
                "n_duplicate_pairs": len(edge_pairs) - len(set(edge_pairs)),
            }
        )
    return report


class PocketGraphDataset(Dataset):
    def __init__(
        self,
        pockets: list[PocketRecord],
        esm_dim: int,
        edge_radius: float = DEFAULT_EDGE_RADIUS,
        normalization_stats: FeatureNormalizationStats | None = None,
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
        pockets: list[PocketRecord],
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


__all__ = [
    "FeatureNormalizationStats",
    "PocketGraphDataset",
    "apply_feature_normalization",
    "compute_feature_normalization_stats",
    "summarize_graph_dataset",
]
