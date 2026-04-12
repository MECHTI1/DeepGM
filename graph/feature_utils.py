from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import Tensor

from data_structures import PocketRecord


def attach_esm_embeddings(
    pocket: PocketRecord,
    esm_lookup: Dict[Tuple[str, int, str], Tensor],
    esm_dim: int,
    zero_if_missing: bool = True,
) -> None:
    for residue in pocket.residues:
        key = residue.residue_id()
        if key in esm_lookup:
            residue.esm_embedding = esm_lookup[key].float()
            residue.has_esm_embedding = True
            continue

        if not zero_if_missing:
            raise KeyError(f"Missing ESM embedding for residue key {key}")

        residue.esm_embedding = torch.zeros(esm_dim, dtype=torch.float32)
        residue.has_esm_embedding = False


def attach_external_residue_features(
    pocket: PocketRecord,
    feature_lookup: Dict[Tuple[str, int, str], Dict[str, float]],
    strict: bool = False,
) -> None:
    for residue in pocket.residues:
        key = residue.residue_id()
        if key in feature_lookup:
            residue.external_features.update(feature_lookup[key])
            residue.has_external_features = True
            continue

        if strict:
            raise KeyError(f"Missing external feature dict for residue key {key}")
        residue.has_external_features = False


__all__ = [
    "attach_esm_embeddings",
    "attach_external_residue_features",
]
