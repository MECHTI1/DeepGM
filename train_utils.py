from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from data_structures import (
    AA_ORDER,
    BACKBONE_ATOMS,
    DEFAULT_EDGE_RADIUS,
    EDGE_SOURCE_TO_INDEX,
    NORMALIZABLE_FEATURE_NAMES,
    PocketRecord,
    ResidueRecord,
)
from featurization import donor_atom_names
from graph_construction import pocket_to_pyg_data
from model import GVPPocketClassifier


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
) -> List[Dict[str, Any]]:
    report: List[Dict[str, Any]] = []
    ring_idx = EDGE_SOURCE_TO_INDEX["ring"]

    for pocket in pockets:
        data = pocket_to_pyg_data(pocket, esm_dim=esm_dim, edge_radius=edge_radius)
        edge_pairs = list(zip(data.edge_index[0].tolist(), data.edge_index[1].tolist()))
        duplicate_pairs = len(edge_pairs) - len(set(edge_pairs))
        ring_mask = data.edge_source_type[:, ring_idx] > 0.5
        report.append(
            {
                "pocket_id": pocket.pocket_id,
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
    ):
        self.pockets = pockets
        self.esm_dim = esm_dim
        self.edge_radius = edge_radius
        self.normalization_stats = normalization_stats

    @classmethod
    def fit_normalization_stats(
        cls,
        pockets: List[PocketRecord],
        esm_dim: int,
        edge_radius: float = DEFAULT_EDGE_RADIUS,
        clamp_value: float = 5.0,
    ) -> FeatureNormalizationStats:
        data_list = [
            pocket_to_pyg_data(pocket, esm_dim=esm_dim, edge_radius=edge_radius)
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
    pred = logits.argmax(dim=-1)
    return float((pred == y).float().mean().item())


def random_residue(resname: Optional[str] = None, center: Optional[Tensor] = None) -> ResidueRecord:
    if resname is None:
        resname = random.choice(AA_ORDER)
    if center is None:
        center = torch.randn(3) * 3.0

    atoms = {
        "CA": center + torch.randn(3) * 0.2,
        "N": center + torch.tensor([-1.2, 0.3, 0.0]) + torch.randn(3) * 0.1,
        "C": center + torch.tensor([1.2, -0.2, 0.0]) + torch.randn(3) * 0.1,
        "O": center + torch.tensor([1.8, -0.5, 0.1]) + torch.randn(3) * 0.1,
    }

    for atom_name in donor_atom_names(resname):
        atoms[atom_name] = center + torch.randn(3) * 0.6 + torch.tensor([0.5, 0.5, 0.5])

    if all(atom_name in BACKBONE_ATOMS for atom_name in atoms.keys()):
        atoms["CB"] = center + torch.tensor([0.3, 1.1, 0.2]) + torch.randn(3) * 0.1

    rr = ResidueRecord(
        chain_id="A",
        resseq=random.randint(1, 999),
        icode="",
        resname=resname,
        atoms=atoms,
    )
    rr.external_features = {
        "SASA": random.uniform(0.0, 100.0),
        "BSA": random.uniform(0.0, 100.0),
        "SolvEnergy": random.uniform(-5.0, 5.0),
        "fa_sol": random.uniform(-3.0, 3.0),
        "fa_elec": random.uniform(-3.0, 3.0),
        "pKa_shift": random.uniform(-4.0, 4.0),
        "dpKa_desolv": random.uniform(-4.0, 4.0),
        "dpKa_bg": random.uniform(-4.0, 4.0),
        "dpKa_titr": random.uniform(-4.0, 4.0),
        "omega": random.uniform(-2.0, 2.0),
        "rama_prepro": random.uniform(-2.0, 2.0),
        "fa_dun": random.uniform(-2.0, 2.0),
        "fa_atr": random.uniform(-3.0, 0.0),
        "fa_rep": random.uniform(0.0, 3.0),
    }
    return rr


def synthetic_pocket(
    pocket_id: str,
    n_residues: int,
    esm_dim: int,
    n_metal_classes: int = 8,
    n_ec_classes: int = 7,
) -> PocketRecord:
    metal = torch.zeros(3, dtype=torch.float32)

    residues = []
    for _ in range(n_residues):
        center = torch.randn(3) * 3.0
        rr = random_residue(center=center)
        rr.esm_embedding = torch.randn(esm_dim)
        residues.append(rr)

    return PocketRecord(
        structure_id="synthetic",
        pocket_id=pocket_id,
        metal_element="ZN",
        metal_coord=metal,
        residues=residues,
        y_metal=random.randint(0, n_metal_classes - 1),
        y_ec=random.randint(0, n_ec_classes - 1),
    )


def run_smoke_test(device: str = "cpu") -> None:
    torch.manual_seed(0)
    random.seed(0)

    esm_dim = 256
    pockets = [
        synthetic_pocket("p0", n_residues=14, esm_dim=esm_dim),
        synthetic_pocket("p1", n_residues=18, esm_dim=esm_dim),
        synthetic_pocket("p2", n_residues=11, esm_dim=esm_dim),
        synthetic_pocket("p3", n_residues=20, esm_dim=esm_dim),
    ]

    graph_summary = summarize_graph_dataset(pockets, esm_dim=esm_dim, edge_radius=10.0)
    normalization_stats = PocketGraphDataset.fit_normalization_stats(
        pockets,
        esm_dim=esm_dim,
        edge_radius=10.0,
        clamp_value=5.0,
    )
    metal_class_weights, ec_class_weights = balanced_class_weights_from_pockets(
        pockets,
        n_metal_classes=8,
        n_ec_classes=7,
    )

    dataset = PocketGraphDataset(
        pockets,
        esm_dim=esm_dim,
        edge_radius=10.0,
        normalization_stats=normalization_stats,
    )
    loader = DataLoader(dataset, batch_size=2, shuffle=True)

    model = GVPPocketClassifier(
        esm_dim=esm_dim,
        hidden_s=128,
        hidden_v=16,
        edge_hidden=64,
        n_layers=4,
        n_metal=8,
        n_ec=7,
        esm_fusion_dim=128,
        metal_class_weights=metal_class_weights,
        ec_class_weights=ec_class_weights,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    loss = train_epoch(model, loader, optimizer, device=device)
    print(f"Smoke-test train loss: {loss:.4f}")

    result = predict_batch(model, loader, device=device)
    print("Graph summary sample:", graph_summary[0])
    print("Metal logits shape:", tuple(result["metal_logits"].shape))
    print("EC logits shape:", tuple(result["ec_logits"].shape))
    print("Metal acc (random synthetic):", accuracy_from_logits(result["metal_logits"], result["metal_y"]))
    print("EC acc (random synthetic):", accuracy_from_logits(result["ec_logits"], result["ec_y"]))
    print("Smoke test completed successfully.")

