from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from Bio.PDB import MMCIFParser, PDBParser
from torch import Tensor

from data_structures import (
    AA_ORDER,
    DEFAULT_MULTINUCLEAR_MERGE_DISTANCE,
    DEFAULT_POCKET_RADIUS,
    GENERIC_METAL_ELEMENT,
    PocketRecord,
    ResidueRecord,
    SUPPORTED_SITE_METAL_ELEMENTS,
)
from featurization import pairwise_distances, safe_norm


@dataclass(frozen=True)
class MetalAtomRecord:
    coord: Tensor
    symbol: str
    site_id: Tuple[str, int, str]


def parse_structure_file(filepath: str, structure_id: Optional[str] = None):
    path = Path(filepath)
    sid = structure_id or path.stem
    parser = MMCIFParser(QUIET=True) if path.suffix.lower() in {".cif", ".mmcif"} else PDBParser(QUIET=True)
    return parser.get_structure(sid, str(path))


def residue_record_from_biopython_residue(residue) -> Optional[ResidueRecord]:
    hetflag, resseq, icode = residue.id
    if hetflag.strip() not in {"", "W"} and residue.resname.strip() not in AA_ORDER:
        return None

    atoms = {}
    for atom in residue.get_atoms():
        atoms[atom.get_name().strip()] = torch.tensor(atom.coord, dtype=torch.float32)

    if "CA" not in atoms:
        return None

    return ResidueRecord(
        chain_id=str(residue.get_parent().id),
        resseq=int(resseq),
        icode=str(icode).strip() if str(icode).strip() else "",
        resname=residue.resname.strip(),
        atoms=atoms,
    )


def normalize_site_metal_resname(resname: str) -> str:
    return "".join(ch for ch in resname.strip().upper() if ch.isalpha())


def canonicalize_site_metal_resname(resname: str) -> Optional[str]:
    raw = resname.strip().upper()
    letters_only = normalize_site_metal_resname(raw)

    for symbol in sorted(SUPPORTED_SITE_METAL_ELEMENTS, key=len, reverse=True):
        if symbol in raw and letters_only == symbol:
            return symbol
    return None


def biopython_residue_site_id(residue, chain_id: str) -> Tuple[str, int, str]:
    _, resseq, icode = residue.id
    return str(chain_id), int(resseq), str(icode).strip() if str(icode).strip() else ""


def metal_records_from_biopython_residue(residue, chain_id: str) -> List[MetalAtomRecord]:
    metal_symbol = canonicalize_site_metal_resname(residue.resname)
    if metal_symbol is None:
        return []

    site_id = biopython_residue_site_id(residue, chain_id)
    return [
        MetalAtomRecord(
            coord=torch.tensor(atom.coord, dtype=torch.float32),
            symbol=metal_symbol,
            site_id=site_id,
        )
        for atom in residue.get_atoms()
    ]


def collect_structure_residues_and_metals(
    structure,
) -> Tuple[List[ResidueRecord], List[MetalAtomRecord]]:
    all_residues: List[ResidueRecord] = []
    metal_records: List[MetalAtomRecord] = []

    for model in structure:
        for chain in model:
            for residue in chain:
                residue_metal_records = metal_records_from_biopython_residue(residue, chain.id)
                if residue_metal_records:
                    metal_records.extend(residue_metal_records)
                    continue

                residue_record = residue_record_from_biopython_residue(residue)
                if residue_record is not None:
                    all_residues.append(residue_record)

    return all_residues, metal_records


def cluster_metal_records(
    metal_records: List[MetalAtomRecord],
    merge_distance: float,
) -> List[List[MetalAtomRecord]]:
    if not metal_records:
        return []

    coords = torch.stack([record.coord.float() for record in metal_records], dim=0)
    dmat = pairwise_distances(coords)

    clusters: List[List[MetalAtomRecord]] = []
    visited = set()
    for start_idx in range(len(metal_records)):
        if start_idx in visited:
            continue

        stack = [start_idx]
        component = []
        visited.add(start_idx)
        while stack:
            idx = stack.pop()
            component.append(metal_records[idx])
            neighbors = torch.where(dmat[idx] <= merge_distance)[0].tolist()
            for neighbor_idx in neighbors:
                if neighbor_idx in visited:
                    continue
                visited.add(neighbor_idx)
                stack.append(neighbor_idx)

        clusters.append(component)

    return clusters


def find_pocket_residues_near_metal_cluster(
    all_residues: List[ResidueRecord],
    metal_cluster: List[MetalAtomRecord],
    pocket_radius: float,
) -> List[ResidueRecord]:
    cluster_tensor = torch.stack([record.coord.float() for record in metal_cluster], dim=0)
    pocket_residues = []

    for residue_record in all_residues:
        residue_coords = torch.stack(list(residue_record.atoms.values()), dim=0)
        diff = residue_coords[:, None, :] - cluster_tensor[None, :, :]
        if safe_norm(diff, dim=-1).min().item() <= pocket_radius:
            pocket_residues.append(residue_record)

    return pocket_residues


def pocket_record_from_metal_cluster(
    structure_id: str,
    cluster_index: int,
    metal_cluster: List[MetalAtomRecord],
    all_residues: List[ResidueRecord],
    pocket_radius: float,
) -> Optional[PocketRecord]:
    pocket_residues = find_pocket_residues_near_metal_cluster(
        all_residues,
        metal_cluster,
        pocket_radius=pocket_radius,
    )
    if not pocket_residues:
        return None

    cluster_coords = [record.coord.float() for record in metal_cluster]
    cluster_symbols = sorted({record.symbol for record in metal_cluster})
    return PocketRecord(
        structure_id=structure_id,
        pocket_id=f"{structure_id}_METAL_{cluster_index}",
        metal_element=cluster_symbols[0] if len(cluster_symbols) == 1 else GENERIC_METAL_ELEMENT,
        metal_coords=cluster_coords,
        residues=pocket_residues,
        metadata={
            "metal_symbols_observed": cluster_symbols,
            "metal_site_ids": [record.site_id for record in metal_cluster],
            "metal_site_coord_map": {
                record.site_id: record.coord.float()
                for record in metal_cluster
            },
        },
    )


def extract_metal_pockets_from_structure(
    structure,
    structure_id: Optional[str] = None,
    pocket_radius: float = DEFAULT_POCKET_RADIUS,
    multinuclear_merge_distance: float = DEFAULT_MULTINUCLEAR_MERGE_DISTANCE,
) -> List[PocketRecord]:
    sid = structure_id or getattr(structure, "id", "unknown_structure")
    all_residues, metal_records = collect_structure_residues_and_metals(structure)
    metal_clusters = cluster_metal_records(metal_records, merge_distance=multinuclear_merge_distance)

    pockets: List[PocketRecord] = []
    for idx, metal_cluster in enumerate(metal_clusters):
        pocket = pocket_record_from_metal_cluster(
            structure_id=sid,
            cluster_index=idx,
            metal_cluster=metal_cluster,
            all_residues=all_residues,
            pocket_radius=pocket_radius,
        )
        if pocket is not None:
            pockets.append(pocket)
    return pockets
