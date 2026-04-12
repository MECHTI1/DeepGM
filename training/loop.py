from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.loader import DataLoader

from data_structures import PocketRecord


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str = "cpu",
) -> float:
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
    return float(evaluate_epoch_with_predictions(model, loader, device=device)["loss"])


def balanced_class_weights_from_labels(labels: list[int], n_classes: int) -> Tensor:
    counts = torch.bincount(torch.tensor(labels, dtype=torch.long), minlength=n_classes).float()
    counts = torch.where(counts > 0, counts, torch.ones_like(counts))
    weights = counts.sum() / (counts * float(n_classes))
    return weights / weights.mean()


def balanced_class_weights_from_pockets(
    pockets: list[PocketRecord],
    n_metal_classes: int,
    n_ec_classes: int,
) -> tuple[Tensor, Tensor]:
    metal_labels = [int(pocket.y_metal) for pocket in pockets if pocket.y_metal is not None]
    ec_labels = [int(pocket.y_ec) for pocket in pockets if pocket.y_ec is not None]
    metal_weights = balanced_class_weights_from_labels(metal_labels, n_metal_classes)
    ec_weights = balanced_class_weights_from_labels(ec_labels, n_ec_classes)
    return metal_weights, ec_weights


@torch.no_grad()
def predict_batch(model: nn.Module, loader: DataLoader, device: str = "cpu") -> dict[str, Tensor]:
    result = evaluate_epoch_with_predictions(model, loader, device=device)
    result.pop("loss", None)
    return result


@torch.no_grad()
def evaluate_epoch_with_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: str = "cpu",
) -> dict[str, Tensor | float]:
    model.eval()
    metal_logits_all = []
    ec_logits_all = []
    metal_y_all = []
    ec_y_all = []
    total = 0.0

    for batch in loader:
        batch = batch.to(device)
        model_outputs = model(batch)
        total += float(model_outputs["loss"].item())
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
    result["loss"] = total / max(1, len(loader))
    return result


@torch.no_grad()
def accuracy_from_logits(logits: Tensor, y: Tensor) -> float:
    pred = logits.argmax(dim=-1)
    return float((pred == y).float().mean().item())


__all__ = [
    "accuracy_from_logits",
    "balanced_class_weights_from_labels",
    "balanced_class_weights_from_pockets",
    "evaluate_epoch_with_predictions",
    "evaluate_epoch",
    "predict_batch",
    "train_epoch",
]
