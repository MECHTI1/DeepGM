from __future__ import annotations

import csv
from typing import Any

import torch

from data_structures import (
    EDGE_SOURCE_TO_INDEX,
    EDGE_SOURCE_TYPES,
    INTERACTION_SUMMARIES_OPTIONAL_WITH_RING,
    PocketRecord,
    RING_INTERACTION_TO_INDEX,
    RING_INTERACTION_TYPES,
)
from featurization import one_hot_index, safe_norm
from graph.edge_geometry import (
    build_pair_edge_geometry,
    candidate_residue_pairs_within_radius,
    canonicalize_edge_pair,
    closest_points_between_residues,
    residue_atom_coords_list,
)
from graph.ring_edges import (
    parse_ring_node_id,
    resolve_ring_edges_path,
    resolve_ring_endpoint_coord,
    ring_edges_path_candidates,
)

RING_METAL_CONTACT_POLICY = "self_loop"


def build_radius_edge_records_from_residues(
    pocket: PocketRecord,
    radius: float,
) -> list[dict[str, Any]]:
    atom_coords_by_residue = residue_atom_coords_list(pocket.residues)
    edge_records: list[dict[str, Any]] = []
    for src_idx, dst_idx in candidate_residue_pairs_within_radius(
        pocket.residues,
        radius,
        atom_coords_by_residue=atom_coords_by_residue,
    ):
        src_residue = pocket.residues[src_idx]
        dst_residue = pocket.residues[dst_idx]
        src_coord, dst_coord, contact_distance = closest_points_between_residues(
            src_residue,
            dst_residue,
            src_coords=atom_coords_by_residue[src_idx],
            dst_coords=atom_coords_by_residue[dst_idx],
        )
        if contact_distance > radius:
            continue
        src_idx, dst_idx, src_coord, dst_coord = canonicalize_edge_pair(src_idx, dst_idx, src_coord, dst_coord)
        src_residue = pocket.residues[src_idx]
        dst_residue = pocket.residues[dst_idx]
        edge_dist_raw, edge_seqsep, edge_same_chain, vector_raw = build_pair_edge_geometry(
            src_residue,
            dst_residue,
            src_coord=src_coord,
            dst_coord=dst_coord,
        )
        edge_records.append(
            {
                "src": src_idx,
                "dst": dst_idx,
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
    return edge_records


def build_ring_interaction_edge_records(
    pocket: PocketRecord,
    require_ring_edges: bool = False,
) -> list[dict[str, Any]]:
    ring_edges_path = resolve_ring_edges_path(pocket)
    if ring_edges_path is None:
        if require_ring_edges:
            raise FileNotFoundError(
                f"Missing RING edge file for pocket {pocket.pocket_id}. "
                f"Tried: {[str(path) for path in ring_edges_path_candidates(pocket.structure_id, pocket.metadata.get('source_path'), pocket.metadata.get('ring_edges_path'), pocket.metadata.get('ring_edges_expected_path'))]}"
            )
        return []

    residue_to_index = {residue.residue_id(): idx for idx, residue in enumerate(pocket.residues)}
    metal_site_coord_map = pocket.metadata.get("metal_site_coord_map", {})
    edge_records: list[dict[str, Any]] = []
    seen_keys: set[tuple[int, int, str]] = set()

    def append_edge_record(
        *,
        src_idx: int,
        dst_idx: int,
        interaction: str,
        src_coord,
        dst_coord,
    ) -> None:
        src_idx, dst_idx, src_coord, dst_coord = canonicalize_edge_pair(src_idx, dst_idx, src_coord, dst_coord)
        edge_key = (src_idx, dst_idx, interaction)
        if edge_key in seen_keys:
            return
        seen_keys.add(edge_key)
        src_residue = pocket.residues[src_idx]
        dst_residue = pocket.residues[dst_idx]
        edge_dist_raw, edge_seqsep, edge_same_chain, vector_raw = build_pair_edge_geometry(
            src_residue,
            dst_residue,
            src_coord=src_coord,
            dst_coord=dst_coord,
        )
        edge_records.append(
            {
                "src": src_idx,
                "dst": dst_idx,
                "dist_raw": edge_dist_raw,
                "seqsep": edge_seqsep,
                "same_chain": edge_same_chain,
                "vector_raw": vector_raw,
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

    with ring_edges_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            interaction = row.get("Interaction", "").strip().upper()
            if interaction not in RING_INTERACTION_TYPES:
                continue

            try:
                src_node = parse_ring_node_id(row["NodeId1"])
                dst_node = parse_ring_node_id(row["NodeId2"])
            except (KeyError, ValueError):
                continue

            src_key = src_node[:3]
            dst_key = dst_node[:3]
            src_is_residue = src_key in residue_to_index
            dst_is_residue = dst_key in residue_to_index
            src_is_metal = src_key in metal_site_coord_map
            dst_is_metal = dst_key in metal_site_coord_map

            if src_is_residue and dst_is_residue:
                src_idx = residue_to_index[src_key]
                dst_idx = residue_to_index[dst_key]
                src_coord = resolve_ring_endpoint_coord(pocket.residues[src_idx], row.get("Atom1", ""))
                dst_coord = resolve_ring_endpoint_coord(pocket.residues[dst_idx], row.get("Atom2", ""))
                if src_coord is None or dst_coord is None:
                    continue
                if float(safe_norm((dst_coord - src_coord).float(), dim=-1).item()) <= 1e-8:
                    continue
                append_edge_record(
                    src_idx=src_idx,
                    dst_idx=dst_idx,
                    interaction=interaction,
                    src_coord=src_coord,
                    dst_coord=dst_coord,
                )
                continue

            if interaction != "METAL_ION:SC_LIG":
                continue
            if RING_METAL_CONTACT_POLICY != "self_loop":
                continue

            if src_is_residue and dst_is_metal:
                residue_idx = residue_to_index[src_key]
                residue = pocket.residues[residue_idx]
                residue_coord = resolve_ring_endpoint_coord(residue, row.get("Atom1", ""))
                metal_coord = metal_site_coord_map.get(dst_key)
                if metal_coord is None or residue_coord is None:
                    continue
                append_edge_record(
                    src_idx=residue_idx,
                    dst_idx=residue_idx,
                    interaction=interaction,
                    src_coord=residue_coord,
                    dst_coord=metal_coord.float(),
                )
                continue

            if dst_is_residue and src_is_metal:
                residue_idx = residue_to_index[dst_key]
                residue = pocket.residues[residue_idx]
                residue_coord = resolve_ring_endpoint_coord(residue, row.get("Atom2", ""))
                metal_coord = metal_site_coord_map.get(src_key)
                if metal_coord is None or residue_coord is None:
                    continue
                append_edge_record(
                    src_idx=residue_idx,
                    dst_idx=residue_idx,
                    interaction=interaction,
                    src_coord=residue_coord,
                    dst_coord=metal_coord.float(),
                )

    return edge_records


def radius_edge_records_from_index(
    pocket: PocketRecord,
    base_edge_index,
) -> list[dict[str, Any]]:
    edge_records: list[dict[str, Any]] = []
    if base_edge_index.size(1) == 0:
        return edge_records

    atom_coords_by_residue = residue_atom_coords_list(pocket.residues)
    seen_pairs: set[tuple[int, int]] = set()
    src_indices, dst_indices = base_edge_index
    for src_idx, dst_idx in zip(src_indices.tolist(), dst_indices.tolist()):
        src_idx, dst_idx, _src_coord, _dst_coord = canonicalize_edge_pair(src_idx, dst_idx)
        edge_key = (src_idx, dst_idx)
        if edge_key in seen_pairs:
            continue
        seen_pairs.add(edge_key)
        src_residue = pocket.residues[src_idx]
        dst_residue = pocket.residues[dst_idx]
        src_coord, dst_coord, _contact_distance = closest_points_between_residues(
            src_residue,
            dst_residue,
            src_coords=atom_coords_by_residue[src_idx],
            dst_coords=atom_coords_by_residue[dst_idx],
        )
        edge_dist_raw, edge_seqsep, edge_same_chain, vector_raw = build_pair_edge_geometry(
            src_residue,
            dst_residue,
            src_coord=src_coord,
            dst_coord=dst_coord,
        )
        edge_records.append(
            {
                "src": src_idx,
                "dst": dst_idx,
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
    return edge_records
