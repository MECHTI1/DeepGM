from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from torch import Tensor

from training.labels import parse_structure_identity


ESM_FILE_RE = re.compile(r"^(?P<structure_id>.+)_chain_(?P<chain>[^_]+)_esmc\.pt$")
ResidueKey = Tuple[str, int, str]


def normalize_chain_id(chain_id: str) -> str:
    normalized = str(chain_id).strip()
    return normalized if normalized else "_"


def normalize_icode(icode: Any) -> str:
    raw = str(icode).strip()
    return raw if raw else ""


def residue_keys_for_structure_chain(structure, chain_id: str) -> Tuple[List[ResidueKey], List[ResidueKey]]:
    wanted_chain = normalize_chain_id(chain_id)
    first_model = next(structure.get_models())
    residue_keys: List[ResidueKey] = []
    residue_keys_with_ca: List[ResidueKey] = []

    for chain in first_model:
        current_chain = normalize_chain_id(chain.id)
        if current_chain != wanted_chain:
            continue
        for residue in chain:
            if residue.id[0] != " ":
                continue
            _, resseq, icode = residue.id
            key = (current_chain, int(resseq), normalize_icode(icode))
            residue_keys.append(key)
            if residue.has_id("CA"):
                residue_keys_with_ca.append(key)

    if not residue_keys:
        raise ValueError(f"Could not find chain {wanted_chain!r} in the parsed structure.")
    return residue_keys, residue_keys_with_ca


def deserialize_residue_ids(raw_residue_ids: List[Any]) -> List[ResidueKey]:
    residue_ids: List[ResidueKey] = []
    for item in raw_residue_ids:
        if isinstance(item, dict):
            chain_id = item.get("chain_id")
            resseq = item.get("resseq")
            icode = item.get("icode", "")
        elif isinstance(item, (list, tuple)) and len(item) == 3:
            chain_id, resseq, icode = item
        else:
            raise ValueError(f"Unsupported residue id entry in embedding payload: {item!r}")
        residue_ids.append((normalize_chain_id(chain_id), int(resseq), normalize_icode(icode)))
    return residue_ids


def serialize_residue_ids(residue_ids: List[ResidueKey]) -> List[dict[str, Any]]:
    return [
        {
            "chain_id": normalize_chain_id(chain_id),
            "resseq": int(resseq),
            "icode": normalize_icode(icode),
        }
        for chain_id, resseq, icode in residue_ids
    ]


def build_embedding_payload(
    embeddings: Tensor,
    residue_ids: List[ResidueKey],
    *,
    structure_id: str | None = None,
    chain_id: str | None = None,
    source_path: str | None = None,
) -> dict[str, Any]:
    if embeddings.dim() != 2:
        raise ValueError(f"Expected a 2D embeddings tensor, got shape {tuple(embeddings.shape)}.")
    if embeddings.size(0) != len(residue_ids):
        raise ValueError(
            f"Embedding payload row count mismatch: got {embeddings.size(0)} rows for {len(residue_ids)} residue ids."
        )
    return {
        "format_version": 2,
        "structure_id": structure_id,
        "chain_id": normalize_chain_id(chain_id) if chain_id is not None else None,
        "source_path": source_path,
        "residue_ids": serialize_residue_ids(residue_ids),
        "embeddings": embeddings.float().cpu(),
    }


def embedding_tensor_and_keys_from_payload(
    payload: Any,
    *,
    structure,
    candidate_path: Path,
    fallback_chain_id: str,
) -> Tuple[Tensor, List[ResidueKey]]:
    if isinstance(payload, dict):
        if "embeddings" not in payload:
            raise ValueError(f"Embedding payload {candidate_path} is missing an 'embeddings' tensor.")
        raw_residue_ids = payload.get("residue_ids")
        if raw_residue_ids is None:
            raise ValueError(f"Embedding payload {candidate_path} is missing 'residue_ids'.")
        return payload["embeddings"].float(), deserialize_residue_ids(list(raw_residue_ids))

    if not isinstance(payload, torch.Tensor):
        raise ValueError(f"Unsupported embedding payload type {type(payload)!r} in {candidate_path}.")

    payload_match = ESM_FILE_RE.match(candidate_path.name)
    chain_id = payload_match.group("chain") if payload_match is not None else fallback_chain_id
    residue_keys, residue_keys_with_ca = residue_keys_for_structure_chain(structure, chain_id)
    if payload.size(0) == len(residue_keys):
        return payload.float(), residue_keys
    if payload.size(0) == len(residue_keys_with_ca):
        return payload.float(), residue_keys_with_ca

    raise ValueError(
        f"Embedding length mismatch for {candidate_path}: got {payload.size(0)} rows, "
        f"expected {len(residue_keys)} chain residues or {len(residue_keys_with_ca)} CA residues."
    )


def embedding_path_candidates(embeddings_dir: Path, structure_path: Path) -> List[Path]:
    candidates: List[Path] = []
    seen = set()

    def add_candidate(path: Path) -> None:
        key = str(path)
        if key in seen:
            return
        seen.add(key)
        candidates.append(path)

    try:
        _pdbid, chain_id, _ec = parse_structure_identity(structure_path.stem)
        normalized_chain_ids = [normalize_chain_id(chain_id)]
    except ValueError:
        normalized_chain_ids = []

    for chain_id in normalized_chain_ids:
        add_candidate(embeddings_dir / f"{structure_path.stem}_chain_{chain_id}_esmc.pt")
        add_candidate(embeddings_dir / structure_path.stem / f"{structure_path.stem}_chain_{chain_id}_esmc.pt")

    add_candidate(embeddings_dir / f"{structure_path.stem}_esmc.pt")
    add_candidate(embeddings_dir / structure_path.stem / f"{structure_path.stem}_esmc.pt")

    for candidate in sorted(embeddings_dir.glob(f"{structure_path.stem}*_esmc.pt")):
        add_candidate(candidate)

    nested_dir = embeddings_dir / structure_path.stem
    if nested_dir.is_dir():
        for candidate in sorted(nested_dir.glob("*_esmc.pt")):
            add_candidate(candidate)

    return candidates


def load_esm_lookup_for_structure(
    structure,
    structure_path: Path,
    embeddings_dir: Path,
) -> Dict[ResidueKey, Tensor]:
    try:
        _pdbid, default_chain_id, _ec = parse_structure_identity(structure_path.stem)
        fallback_chain_id = normalize_chain_id(default_chain_id)
    except ValueError:
        fallback_chain_id = "_"

    esm_lookup: Dict[ResidueKey, Tensor] = {}
    found_files: List[Path] = []
    for candidate in embedding_path_candidates(embeddings_dir, structure_path):
        if not candidate.is_file():
            continue

        payload = torch.load(candidate, map_location="cpu", weights_only=True)
        embeddings, residue_ids = embedding_tensor_and_keys_from_payload(
            payload,
            structure=structure,
            candidate_path=candidate,
            fallback_chain_id=fallback_chain_id,
        )
        if embeddings.dim() != 2:
            raise ValueError(f"Expected a 2D embedding tensor in {candidate}, got shape {tuple(embeddings.shape)}.")
        if embeddings.size(0) != len(residue_ids):
            raise ValueError(
                f"Embedding payload {candidate} has {embeddings.size(0)} rows for {len(residue_ids)} residue ids."
            )

        overlap = set(esm_lookup).intersection(residue_ids)
        if overlap:
            raise ValueError(f"Duplicate ESM residue ids detected while loading {candidate}: {sorted(overlap)[:5]}")

        for residue_id, embedding in zip(residue_ids, embeddings):
            esm_lookup[residue_id] = embedding.float()
        found_files.append(candidate)

    if not found_files:
        raise FileNotFoundError(f"No ESM embedding file found for {structure_path.stem} under {embeddings_dir}.")
    return esm_lookup


__all__ = [
    "ResidueKey",
    "load_esm_lookup_for_structure",
    "normalize_chain_id",
    "normalize_icode",
    "residue_keys_for_structure_chain",
    "build_embedding_payload",
    "deserialize_residue_ids",
    "serialize_residue_ids",
]
