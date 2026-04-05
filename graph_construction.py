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
    DEFAULT_POCKET_RADIUS,
    DEFAULT_SECOND_SHELL_CUTOFF,
    EDGE_SOURCE_TO_INDEX,
    EDGE_SOURCE_TYPES,
    INTERACTION_SUMMARIES_OPTIONAL_WITH_RING,
    PocketRecord,
    RING_INTERACTION_TO_INDEX,
    RING_INTERACTION_TYPES,
    ResidueRecord,
)
from featurization import (
    build_external_feature_groups,
    compute_net_ligand_vector,
    donor_coords_and_mask,
    functional_group_centroid,
    min_distance_to_point,
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


def extract_zn_pockets_from_structure(
    structure,
    structure_id: Optional[str] = None,
    pocket_radius: float = DEFAULT_POCKET_RADIUS,
) -> List[PocketRecord]:
    sid = structure_id or getattr(structure, "id", "unknown_structure")

    all_residues: List[ResidueRecord] = []
    zn_coords: List[Tensor] = []

    for model in structure:
        for chain in model:
            for residue in chain:
                is_zn_residue = residue.resname.strip().upper() in {"ZN", "ZN2", "ZN+2"}
                if is_zn_residue:
                    for atom in residue.get_atoms():
                        zn_coords.append(torch.tensor(atom.coord, dtype=torch.float32))
                    continue

                rr = residue_record_from_biopython_residue(residue)
                if rr is not None:
                    all_residues.append(rr)

    pockets: List[PocketRecord] = []
    for idx, zn in enumerate(zn_coords):
        pocket_residues = []
        for rr in all_residues:
            coords = torch.stack(list(rr.atoms.values()), dim=0)
            min_d = safe_norm(coords - zn.unsqueeze(0), dim=-1).min().item()
            if min_d <= pocket_radius:
                pocket_residues.append(rr)

        if pocket_residues:
            pockets.append(
                PocketRecord(
                    structure_id=sid,
                    pocket_id=f"{sid}_ZN_{idx}",
                    metal_element="ZN",
                    metal_coord=zn,
                    residues=pocket_residues,
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
    second_shell_cutoff: float = DEFAULT_SECOND_SHELL_CUTOFF,
) -> None:
    metal = pocket.metal_coord.float()

    fg_centroids = []
    for rr in pocket.residues:
        donor_coords, donor_mask = donor_coords_and_mask(rr, max_donors=2)
        min_d = min_distance_to_point(donor_coords, metal, donor_mask)
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


def build_radius_graph(pos: Tensor, radius: float) -> Tensor:
    dmat = pairwise_distances(pos)
    mask = (dmat <= radius) & (dmat > 0.0)

    src_list = []
    dst_list = []
    for i in range(pos.size(0)):
        js = torch.where(mask[i])[0]
        if js.numel() == 0:
            continue
        src_list.append(torch.full((js.numel(),), i, dtype=torch.long))
        dst_list.append(js.long())

    if not src_list:
        return torch.zeros(2, 0, dtype=torch.long)

    src = torch.cat(src_list, dim=0)
    dst = torch.cat(dst_list, dim=0)
    return torch.stack([src, dst], dim=0)


def min_fg_fg_distance(rr_i: ResidueRecord, rr_j: ResidueRecord) -> float:
    coords_i, mask_i = donor_coords_and_mask(rr_i, max_donors=2)
    coords_j, mask_j = donor_coords_and_mask(rr_j, max_donors=2)

    if mask_i.any() and mask_j.any():
        ci = coords_i[mask_i]
        cj = coords_j[mask_j]
        diff = ci[:, None, :] - cj[None, :, :]
        return float(safe_norm(diff, dim=-1).min().item())

    fi = functional_group_centroid(rr_i)
    fj = functional_group_centroid(rr_j)
    return float(safe_norm(fi - fj, dim=-1).item())


def build_pair_edge_scalars(rr_i: ResidueRecord, rr_j: ResidueRecord) -> Tuple[Tensor, float, float]:
    ca_i = rr_i.ca()
    ca_j = rr_j.ca()
    if ca_i is None or ca_j is None:
        raise ValueError(f"Missing CA atom for edge pair {rr_i.residue_id()} -> {rr_j.residue_id()}")

    ca_ca_dist = float(safe_norm(ca_j - ca_i, dim=-1).item())
    fg_fg_dist = float(min_fg_fg_distance(rr_i, rr_j))
    edge_dist_raw = torch.tensor([ca_ca_dist, fg_fg_dist], dtype=torch.float32)
    edge_seqsep = float(abs(rr_i.resseq - rr_j.resseq))
    edge_same_chain = float(rr_i.chain_id == rr_j.chain_id)
    return edge_dist_raw, edge_seqsep, edge_same_chain


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

            edge_dist_raw, edge_seqsep, edge_same_chain = build_pair_edge_scalars(src_residue, dst_residue)
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
        residue_to_stage1_node_features(rr, pocket.metal_coord, esm_dim, v_net)
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

    base_edge_index = build_radius_graph(pos, edge_radius)
    ring_edge_records = build_ring_interaction_edge_records(pocket)

    edge_records: List[Dict[str, Any]] = []
    if base_edge_index.size(1) > 0:
        src, dst = base_edge_index
        for i, j in zip(src.tolist(), dst.tolist()):
            rr_i = pocket.residues[i]
            rr_j = pocket.residues[j]
            edge_dist_raw, edge_seqsep, edge_same_chain = build_pair_edge_scalars(rr_i, rr_j)
            edge_records.append(
                {
                    "src": i,
                    "dst": j,
                    "dist_raw": edge_dist_raw,
                    "seqsep": edge_seqsep,
                    "same_chain": edge_same_chain,
                    "vector_raw": (pos[j] - pos[i]).float(),
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
        zinc_pos=pocket.metal_coord.unsqueeze(0),
    )
    if y_metal is not None:
        data.y_metal = y_metal
    if y_ec is not None:
        data.y_ec = y_ec
    return data


def save_pocket_metadata_json(pocket: PocketRecord, outpath: str) -> None:
    payload = {
        "structure_id": pocket.structure_id,
        "pocket_id": pocket.pocket_id,
        "metal_element": pocket.metal_element,
        "metal_coord": pocket.metal_coord.tolist(),
        "y_metal": pocket.y_metal,
        "y_ec": pocket.y_ec,
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

