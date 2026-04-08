from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from Bio.PDB import MMCIFParser, PDBParser
from torch import Tensor
from torch_geometric.data import Data

from data_structures import (
    AA_ORDER,
    DEFAULT_EDGE_RADIUS,
    DEFAULT_FIRST_SHELL_CUTOFF,
    DEFAULT_MULTINUCLEAR_MERGE_DISTANCE,
    DEFAULT_POCKET_RADIUS,
    EDGE_SOURCE_TO_INDEX,
    EDGE_SOURCE_TYPES,
    GENERIC_METAL_ELEMENT,
    INTERACTION_SUMMARIES_OPTIONAL_WITH_RING,
    PocketRecord,
    RING_INTERACTION_TO_INDEX,
    RING_INTERACTION_TYPES,
    ResidueRecord,
    SUPPORTED_SITE_METAL_ELEMENTS,
)
from featurization import (
    compute_net_ligand_vector,
    donor_coords_and_mask,
    functional_group_centroid,
    MultinuclearSiteHandler,
    pairwise_distances,
    residue_to_stage1_node_features,
    safe_norm,
    one_hot_index,
)


def parse_structure_file(filepath: str, structure_id: Optional[str] = None):
    path = Path(filepath)
    sid = structure_id or path.stem

    if path.suffix.lower() in {".cif", ".mmcif"}:
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)

    return parser.get_structure(sid, str(path))


def residue_record_from_biopython_residue(residue) -> Optional[ResidueRecord]:
    hetflag, resseq, icode = residue.id
    if hetflag.strip() not in {"", "W"} and residue.resname.strip() not in AA_ORDER:
        return None

    atoms = {}
    for atom in residue.get_atoms():
        name = atom.get_name().strip()
        coord = torch.tensor(atom.coord, dtype=torch.float32)
        atoms[name] = coord

    if "CA" not in atoms:
        return None

    parent_chain = residue.get_parent().id
    return ResidueRecord(
        chain_id=str(parent_chain),
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

    # Accept cases like FE, FE2, FE3+, ZN2, 2FE, etc., but reject any name that
    # introduces extra alphabetic characters beyond the element symbol itself.
    for symbol in sorted(SUPPORTED_SITE_METAL_ELEMENTS, key=len, reverse=True):
        if symbol in raw and letters_only == symbol:
            return symbol
    return None


def is_supported_site_metal_residue(residue) -> bool:
    return canonicalize_site_metal_resname(residue.resname) is not None


def cluster_metal_records(
    metal_records: List[Dict[str, Any]],
    merge_distance: float,
) -> List[List[Dict[str, Any]]]:
    if not metal_records:
        return []

    coords = torch.stack([record["coord"].float() for record in metal_records], dim=0)
    dmat = pairwise_distances(coords)

    clusters: List[List[Dict[str, Any]]] = []
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


def parse_ring_node_id(node_id: str) -> Tuple[str, int, str, str]:
    parts = node_id.strip().split(":")
    if len(parts) != 4:
        raise ValueError(f"Unsupported ring node id format: {node_id!r}")

    chain_id, resseq_text, icode, resname = parts
    icode = "" if icode in {"_", ".", "?"} else icode
    return chain_id, int(resseq_text), icode, resname


def parse_embedded_coord(text: str) -> Optional[Tensor]:
    raw = text.strip()
    if not raw:
        return None

    parts = [p.strip() for p in raw.split(",")]
    if len(parts) != 3:
        return None

    try:
        values = [float(p) for p in parts]
    except ValueError:
        return None

    return torch.tensor(values, dtype=torch.float32)


def resolve_ring_endpoint_coord(residue: ResidueRecord, atom_or_coord: str) -> Optional[Tensor]:
    coord = parse_embedded_coord(atom_or_coord)
    if coord is not None:
        return coord

    atom_name = atom_or_coord.strip()
    if atom_name:
        atom = residue.get_atom(atom_name)
        if atom is not None:
            return atom.float()
    return None


def resolve_ring_edges_path(pocket: PocketRecord) -> Optional[Path]:
    candidates: List[Path] = []

    explicit = pocket.metadata.get("ring_edges_path")
    if explicit:
        candidates.append(Path(str(explicit)))

    source_path = pocket.metadata.get("source_path")
    if source_path:
        candidates.append(Path(f"{source_path}_ringEdges"))

    structure_id = pocket.structure_id.strip()
    if structure_id:
        emb_dir = Path(__file__).resolve().parent / ".data" / "embeddings" / structure_id
        candidates.append(emb_dir / f"{structure_id}.pdb_ringEdges")
        candidates.append(emb_dir / f"{structure_id}.cif_ringEdges")
        if emb_dir.is_dir():
            candidates.extend(sorted(emb_dir.glob("*ringEdges")))

    seen = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.is_file():
            return candidate
    return None


def extract_metal_pockets_from_structure(
    structure,
    structure_id: Optional[str] = None,
    pocket_radius: float = DEFAULT_POCKET_RADIUS,
    multinuclear_merge_distance: float = DEFAULT_MULTINUCLEAR_MERGE_DISTANCE,
) -> List[PocketRecord]:
    sid = structure_id or getattr(structure, "id", "unknown_structure")

    all_residues: List[ResidueRecord] = []
    metal_records: List[Dict[str, Any]] = []

    for model in structure:
        for chain in model:
            for residue in chain:
                metal_symbol = canonicalize_site_metal_resname(residue.resname)
                if metal_symbol is not None:
                    _, resseq, icode = residue.id
                    site_id = (str(chain.id), int(resseq), str(icode).strip() if str(icode).strip() else "")
                    for atom in residue.get_atoms():
                        metal_records.append(
                            {
                                "coord": torch.tensor(atom.coord, dtype=torch.float32),
                                "symbol": metal_symbol,
                                "site_id": site_id,
                            }
                        )
                    continue

                rr = residue_record_from_biopython_residue(residue)
                if rr is not None:
                    all_residues.append(rr)

    metal_clusters = cluster_metal_records(metal_records, merge_distance=multinuclear_merge_distance)
    pockets: List[PocketRecord] = []
    for idx, metal_cluster in enumerate(metal_clusters):
        cluster_coords = [record["coord"].float() for record in metal_cluster]
        cluster_symbols = sorted({record["symbol"] for record in metal_cluster})
        cluster_tensor = torch.stack(cluster_coords, dim=0)
        cluster_center = cluster_tensor.mean(dim=0)
        pocket_residues = []
        for rr in all_residues:
            coords = torch.stack(list(rr.atoms.values()), dim=0)
            diff = coords[:, None, :] - cluster_tensor[None, :, :]
            min_d = safe_norm(diff, dim=-1).min().item()
            if min_d <= pocket_radius:
                pocket_residues.append(rr)

        if pocket_residues:
            pockets.append(
                PocketRecord(
                    structure_id=sid,
                    pocket_id=f"{sid}_METAL_{idx}",
                    metal_element=cluster_symbols[0] if len(cluster_symbols) == 1 else GENERIC_METAL_ELEMENT,
                    metal_coord=cluster_center,
                    metal_coords=cluster_coords,
                    residues=pocket_residues,
                    y_multinuclear=int(len(cluster_coords) > 1),
                    metadata={
                        "metal_symbols_observed": cluster_symbols,
                        "metal_site_ids": [record["site_id"] for record in metal_cluster],
                        "metal_count": len(cluster_coords),
                        "is_multinuclear": bool(len(cluster_coords) > 1),
                    },
                )
            )
    return pockets


def attach_esm_embeddings(
    pocket: PocketRecord,
    esm_lookup: Dict[Tuple[str, int, str], Tensor],
    esm_dim: int,
    zero_if_missing: bool = True,
) -> None:
    for rr in pocket.residues:
        key = rr.residue_id()
        if key in esm_lookup:
            rr.esm_embedding = esm_lookup[key].float()
        elif zero_if_missing:
            rr.esm_embedding = torch.zeros(esm_dim, dtype=torch.float32)
        else:
            raise KeyError(f"Missing ESM embedding for residue key {key}")


def attach_external_residue_features(
    pocket: PocketRecord,
    feature_lookup: Dict[Tuple[str, int, str], Dict[str, float]],
    strict: bool = False,
) -> None:
    for rr in pocket.residues:
        key = rr.residue_id()
        if key in feature_lookup:
            rr.external_features.update(feature_lookup[key])
        elif strict:
            raise KeyError(f"Missing external feature dict for residue key {key}")


def annotate_shell_roles(
    pocket: PocketRecord,
    first_shell_cutoff: float = DEFAULT_FIRST_SHELL_CUTOFF,
    second_shell_cutoff: float = 4.5,
) -> None:
    metal_coords = MultinuclearSiteHandler.metal_coords_for_pocket(pocket)

    fg_centroids = []
    for rr in pocket.residues:
        donor_coords, donor_mask = donor_coords_and_mask(rr, max_donors=2)
        min_d = MultinuclearSiteHandler.min_distance_to_metals(donor_coords, metal_coords, donor_mask)
        rr.is_first_shell = min_d <= first_shell_cutoff
        rr.is_second_shell = False
        fg_centroids.append(functional_group_centroid(rr))

    first_shell_centroids = [fg for rr, fg in zip(pocket.residues, fg_centroids) if rr.is_first_shell]

    for rr, fg in zip(pocket.residues, fg_centroids):
        if rr.is_first_shell:
            continue
        if len(first_shell_centroids) == 0:
            rr.is_second_shell = False
            continue
        dists = [safe_norm(fg - f0, dim=-1).item() for f0 in first_shell_centroids]
        rr.is_second_shell = min(dists) <= second_shell_cutoff

def residue_atom_coords(rr: ResidueRecord) -> Tensor:
    return torch.stack([coord.float() for coord in rr.atoms.values()], dim=0)


def closest_points_between_residues(rr_i: ResidueRecord, rr_j: ResidueRecord) -> Tuple[Tensor, Tensor, float]:
    coords_i = residue_atom_coords(rr_i)
    coords_j = residue_atom_coords(rr_j)

    diff = coords_i[:, None, :] - coords_j[None, :, :]
    dmat = safe_norm(diff, dim=-1)
    flat_idx = int(torch.argmin(dmat).item())
    i_idx = flat_idx // dmat.size(1)
    j_idx = flat_idx % dmat.size(1)

    src_coord = coords_i[i_idx]
    dst_coord = coords_j[j_idx]
    distance = float(dmat[i_idx, j_idx].item())
    return src_coord, dst_coord, distance


def build_pair_edge_geometry(
    rr_i: ResidueRecord,
    rr_j: ResidueRecord,
    src_coord: Optional[Tensor] = None,
    dst_coord: Optional[Tensor] = None,
) -> Tuple[Tensor, float, float, Tensor]:
    ca_i = rr_i.ca()
    ca_j = rr_j.ca()
    if ca_i is None or ca_j is None:
        raise ValueError(f"Missing CA atom for edge pair {rr_i.residue_id()} -> {rr_j.residue_id()}")

    if src_coord is None or dst_coord is None:
        src_coord, dst_coord, contact_dist = closest_points_between_residues(rr_i, rr_j)
    else:
        src_coord = src_coord.float()
        dst_coord = dst_coord.float()
        contact_dist = float(safe_norm(dst_coord - src_coord, dim=-1).item())

    vector_raw = (dst_coord - src_coord).float()
    ca_ca_dist = float(safe_norm(ca_j - ca_i, dim=-1).item())
    edge_dist_raw = torch.tensor([contact_dist, ca_ca_dist], dtype=torch.float32)
    edge_seqsep = float(abs(rr_i.resseq - rr_j.resseq))
    edge_same_chain = float(rr_i.chain_id == rr_j.chain_id)
    return edge_dist_raw, edge_seqsep, edge_same_chain, vector_raw


def build_radius_graph_from_residues(residues: List[ResidueRecord], radius: float) -> Tensor:
    src_list = []
    dst_list = []

    for i, rr_i in enumerate(residues):
        for j, rr_j in enumerate(residues):
            if i == j:
                continue
            _, _, contact_dist = closest_points_between_residues(rr_i, rr_j)
            if contact_dist <= radius:
                src_list.append(i)
                dst_list.append(j)

    if not src_list:
        return torch.zeros(2, 0, dtype=torch.long)

    return torch.tensor([src_list, dst_list], dtype=torch.long)


def build_ring_interaction_edge_records(pocket: PocketRecord) -> List[Dict[str, Any]]:
    ring_edges_path = resolve_ring_edges_path(pocket)
    if ring_edges_path is None:
        return []

    residue_to_index = {rr.residue_id(): i for i, rr in enumerate(pocket.residues)}
    records: List[Dict[str, Any]] = []

    with ring_edges_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            interaction = row.get("Interaction", "").strip().upper()
            if interaction not in RING_INTERACTION_TYPES:
                continue

            try:
                src_key = parse_ring_node_id(row["NodeId1"])[:3]
                dst_key = parse_ring_node_id(row["NodeId2"])[:3]
            except (KeyError, ValueError):
                continue

            if src_key not in residue_to_index or dst_key not in residue_to_index:
                continue

            src_idx = residue_to_index[src_key]
            dst_idx = residue_to_index[dst_key]
            src_residue = pocket.residues[src_idx]
            dst_residue = pocket.residues[dst_idx]

            src_coord = resolve_ring_endpoint_coord(src_residue, row.get("Atom1", ""))
            dst_coord = resolve_ring_endpoint_coord(dst_residue, row.get("Atom2", ""))
            if src_coord is None or dst_coord is None:
                continue

            vector = (dst_coord - src_coord).float()
            if float(safe_norm(vector, dim=-1).item()) <= 1e-8:
                continue

            edge_dist_raw, edge_seqsep, edge_same_chain, vector = build_pair_edge_geometry(
                src_residue,
                dst_residue,
                src_coord=src_coord,
                dst_coord=dst_coord,
            )
            records.append(
                {
                    "src": src_idx,
                    "dst": dst_idx,
                    "dist_raw": edge_dist_raw,
                    "seqsep": edge_seqsep,
                    "same_chain": edge_same_chain,
                    "vector_raw": vector,
                    "interaction_type": one_hot_index(
                        RING_INTERACTION_TO_INDEX[interaction],
                        len(INTERACTION_SUMMARIES_OPTIONAL_WITH_RING),
                    ),
                    "source_type": one_hot_index(
                        EDGE_SOURCE_TO_INDEX["ring"],
                        len(EDGE_SOURCE_TYPES),
                    ),
                }
            )

    return records


def pocket_to_pyg_data(
    pocket: PocketRecord,
    esm_dim: int,
    edge_radius: float = DEFAULT_EDGE_RADIUS,
) -> Data:
    annotate_shell_roles(pocket)
    v_net = compute_net_ligand_vector(pocket)

    node_dicts = [
        residue_to_stage1_node_features(rr, pocket, esm_dim, v_net)
        for rr in pocket.residues
    ]

    x_esm = torch.stack([d["x_esm"] for d in node_dicts], dim=0)
    x_reschem = torch.stack([d["x_reschem"] for d in node_dicts], dim=0)
    x_role = torch.stack([d["x_role"] for d in node_dicts], dim=0)
    x_dist_raw = torch.stack([d["x_dist_raw"] for d in node_dicts], dim=0)
    x_misc = torch.stack([d["x_misc"] for d in node_dicts], dim=0)
    x_env_burial = torch.stack([d["x_env_burial"] for d in node_dicts], dim=0)
    x_env_pka = torch.stack([d["x_env_pka"] for d in node_dicts], dim=0)
    x_env_conf = torch.stack([d["x_env_conf"] for d in node_dicts], dim=0)
    x_env_interactions = torch.stack([d["x_env_interactions"] for d in node_dicts], dim=0)
    x_vec = torch.stack([d["x_vec"] for d in node_dicts], dim=0)
    donor_coords = torch.stack([d["donor_coords"] for d in node_dicts], dim=0)
    donor_mask = torch.stack([d["donor_mask"] for d in node_dicts], dim=0)
    fg_centroid = torch.stack([d["fg_centroid"] for d in node_dicts], dim=0)
    pos = torch.stack([d["pos"] for d in node_dicts], dim=0)

    base_edge_index = build_radius_graph_from_residues(pocket.residues, edge_radius)
    ring_edge_records = build_ring_interaction_edge_records(pocket)
    # Keep both local-radius edges and explicit ring-interaction edges; they encode
    # different edge sources for the same local pocket.

    edge_records: List[Dict[str, Any]] = []
    if base_edge_index.size(1) > 0:
        src, dst = base_edge_index
        for i, j in zip(src.tolist(), dst.tolist()):
            rr_i = pocket.residues[i]
            rr_j = pocket.residues[j]
            edge_dist_raw, edge_seqsep, edge_same_chain, vector_raw = build_pair_edge_geometry(rr_i, rr_j)
            edge_records.append(
                {
                    "src": i,
                    "dst": j,
                    "dist_raw": edge_dist_raw,
                    "seqsep": edge_seqsep,
                    "same_chain": edge_same_chain,
                    "vector_raw": vector_raw,
                    "interaction_type": torch.zeros(
                        len(INTERACTION_SUMMARIES_OPTIONAL_WITH_RING),
                        dtype=torch.float32,
                    ),
                    "source_type": one_hot_index(
                        EDGE_SOURCE_TO_INDEX["radius"],
                        len(EDGE_SOURCE_TYPES),
                    ),
                }
            )

    edge_records.extend(ring_edge_records)
    if not edge_records:
        raise ValueError(
            f"Pocket {pocket.pocket_id} produced a graph with no edges at edge_radius={edge_radius}. "
            "Increase the radius, inspect the pocket residues, or provide ring interaction edges."
        )

    edge_index = torch.tensor(
        [
            [rec["src"] for rec in edge_records],
            [rec["dst"] for rec in edge_records],
        ],
        dtype=torch.long,
    )
    edge_dist_raw = torch.stack([rec["dist_raw"] for rec in edge_records], dim=0)
    edge_seqsep = torch.tensor([rec["seqsep"] for rec in edge_records], dtype=torch.float32).unsqueeze(-1)
    edge_same_chain = torch.tensor([rec["same_chain"] for rec in edge_records], dtype=torch.float32).unsqueeze(-1)
    edge_vector_raw = torch.stack([rec["vector_raw"] for rec in edge_records], dim=0)
    edge_interaction_type = torch.stack([rec["interaction_type"] for rec in edge_records], dim=0)
    edge_source_type = torch.stack([rec["source_type"] for rec in edge_records], dim=0)

    y_metal = None if pocket.y_metal is None else torch.tensor([pocket.y_metal], dtype=torch.long)
    y_ec = None if pocket.y_ec is None else torch.tensor([pocket.y_ec], dtype=torch.long)

    data = Data(
        x_esm=x_esm,
        x_reschem=x_reschem,
        x_role=x_role,
        x_dist_raw=x_dist_raw,
        x_misc=x_misc,
        x_env_burial=x_env_burial,
        x_env_pka=x_env_pka,
        x_env_conf=x_env_conf,
        x_env_interactions=x_env_interactions,
        x_vec=x_vec,
        pos=pos,
        fg_centroid=fg_centroid,
        donor_coords=donor_coords,
        donor_mask=donor_mask,
        edge_index=edge_index,
        edge_dist_raw=edge_dist_raw,
        edge_seqsep=edge_seqsep,
        edge_same_chain=edge_same_chain,
        edge_vector_raw=edge_vector_raw,
        edge_interaction_type=edge_interaction_type,
        edge_source_type=edge_source_type,
        metal_pos=MultinuclearSiteHandler.metal_coords_for_pocket(pocket),
        metal_center_pos=pocket.metal_coord.unsqueeze(0),
        metal_count=torch.tensor([pocket.metal_count()], dtype=torch.long),
        is_multinuclear=torch.tensor([int(pocket.is_multinuclear())], dtype=torch.long),
        site_metal_stats=MultinuclearSiteHandler.site_metal_stats(pocket).unsqueeze(0),
    )
    if y_metal is not None:
        data.y_metal = y_metal
    if y_ec is not None:
        data.y_ec = y_ec
    if pocket.y_multinuclear is not None:
        data.y_multinuclear = torch.tensor([pocket.y_multinuclear], dtype=torch.long)
    return data


def save_pocket_metadata_json(pocket: PocketRecord, outpath: str) -> None:
    payload = {
        "structure_id": pocket.structure_id,
        "pocket_id": pocket.pocket_id,
        "metal_element": pocket.metal_element,
        "metal_coord": pocket.metal_coord.tolist(),
        "metal_coords": [coord.tolist() for coord in pocket.resolved_metal_coords()],
        "metal_count": pocket.metal_count(),
        "is_multinuclear": pocket.is_multinuclear(),
        "y_metal": pocket.y_metal,
        "y_ec": pocket.y_ec,
        "y_multinuclear": pocket.y_multinuclear,
        "residues": [
            {
                "chain_id": rr.chain_id,
                "resseq": rr.resseq,
                "icode": rr.icode,
                "resname": rr.resname,
                "is_first_shell": rr.is_first_shell,
                "is_second_shell": rr.is_second_shell,
                "external_features": rr.external_features,
                "atom_names": sorted(list(rr.atoms.keys())),
            }
            for rr in pocket.residues
        ],
    }

    with open(outpath, "w") as handle:
        json.dump(payload, handle, indent=2)
