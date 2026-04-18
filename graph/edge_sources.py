from __future__ import annotations

import csv

import torch

from data_structures import (
    EDGE_SOURCE_TO_INDEX,
    EDGE_SOURCE_TYPES,
    RING_INTERACTION_ALIASES,
    INTERACTION_SUMMARIES_OPTIONAL_WITH_RING,
    PocketRecord,
    RING_INTERACTION_TO_INDEX,
    RING_INTERACTION_TYPES,
)
from featurization import one_hot_index, safe_norm
from graph.edge_geometry import (
    build_pair_edge_geometry,
    build_radius_pair_geometries,
    canonicalize_edge_pair,
)
from graph.edge_records import ResidueEdgeRecord, ResidueMetalEdgeRecord
from graph.ring_edges import (
    parse_ring_node_id,
    resolve_ring_edges_path,
    resolve_ring_endpoint_coord,
    ring_edges_path_candidates,
)


def normalize_ring_interaction_type(interaction: str) -> str | None:
    interaction = interaction.strip().upper()
    interaction = RING_INTERACTION_ALIASES.get(interaction, interaction)
    if interaction not in RING_INTERACTION_TYPES:
        return None
    return interaction


def build_radius_edge_records_from_residues(
    pocket: PocketRecord,
    radius: float,
) -> list[ResidueEdgeRecord]:
    return [
        ResidueEdgeRecord(
            src=geometry.src_idx,
            dst=geometry.dst_idx,
            dist_raw=geometry.dist_raw.clone(),
            seqsep=geometry.seqsep,
            same_chain=geometry.same_chain,
            vector_raw=geometry.vector_raw.clone(),
            interaction_type=torch.zeros(
                len(INTERACTION_SUMMARIES_OPTIONAL_WITH_RING),
                dtype=torch.float32,
            ),
            source_type=one_hot_index(
                EDGE_SOURCE_TO_INDEX["radius"],
                len(EDGE_SOURCE_TYPES),
            ),
            geometry_label="closest_atoms",
        )
        for geometry in build_radius_pair_geometries(pocket.residues, radius)
    ]


def _build_residue_edge_record(
    *,
    src_idx: int,
    dst_idx: int,
    dist_raw,
    seqsep: float,
    same_chain: float,
    vector_raw,
    interaction: str,
    geometry_label: str,
) -> ResidueEdgeRecord:
    return ResidueEdgeRecord(
        src=src_idx,
        dst=dst_idx,
        dist_raw=dist_raw.clone(),
        seqsep=seqsep,
        same_chain=same_chain,
        vector_raw=vector_raw.clone(),
        interaction_type=one_hot_index(
            RING_INTERACTION_TO_INDEX[interaction],
            len(INTERACTION_SUMMARIES_OPTIONAL_WITH_RING),
        ),
        source_type=one_hot_index(
            EDGE_SOURCE_TO_INDEX["ring"],
            len(EDGE_SOURCE_TYPES),
        ),
        geometry_label=geometry_label,
    )


def _build_residue_metal_edge_record(
    *,
    residue_idx: int,
    metal_idx: int,
    residue_coord,
    metal_coord,
    interaction: str,
) -> ResidueMetalEdgeRecord:
    vector_raw = (metal_coord.float() - residue_coord.float()).float()
    contact_distance = float(safe_norm(vector_raw, dim=-1).item())
    return ResidueMetalEdgeRecord(
        residue_idx=residue_idx,
        metal_idx=metal_idx,
        dist_raw=torch.tensor([contact_distance], dtype=torch.float32),
        vector_raw=vector_raw,
        interaction_type=one_hot_index(
            RING_INTERACTION_TO_INDEX[interaction],
            len(INTERACTION_SUMMARIES_OPTIONAL_WITH_RING),
        ),
        source_type=one_hot_index(
            EDGE_SOURCE_TO_INDEX["ring"],
            len(EDGE_SOURCE_TYPES),
        ),
        geometry_label="residue_to_metal",
    )


def _resolve_metal_index(
    pocket: PocketRecord,
    site_key: tuple[str, int, str],
) -> int | None:
    metal_site_ids = pocket.metadata.get("metal_site_ids", [])
    if site_key in metal_site_ids:
        return int(metal_site_ids.index(site_key))

    metal_coord = pocket.metadata.get("metal_site_coord_map", {}).get(site_key)
    if metal_coord is None:
        return None

    metal_coords = torch.stack([coord.float() for coord in pocket.metal_coords], dim=0)
    dists = safe_norm(metal_coords - metal_coord.float().unsqueeze(0), dim=-1)
    return int(torch.argmin(dists).item())


def build_ring_edge_records(
    pocket: PocketRecord,
    require_ring_edges: bool = False,
) -> tuple[list[ResidueEdgeRecord], list[ResidueMetalEdgeRecord]]:
    ring_edges_path = resolve_ring_edges_path(pocket)
    if ring_edges_path is None:
        if require_ring_edges:
            raise FileNotFoundError(
                f"Missing RING edge file for pocket {pocket.pocket_id}. "
                f"Tried: {[str(path) for path in ring_edges_path_candidates(pocket.structure_id, pocket.metadata.get('source_path'), pocket.metadata.get('ring_edges_path'), pocket.metadata.get('ring_edges_expected_path'))]}"
            )
        return [], []

    residue_to_index = {residue.residue_id(): idx for idx, residue in enumerate(pocket.residues)}
    metal_site_coord_map = pocket.metadata.get("metal_site_coord_map", {})
    residue_edge_records: list[ResidueEdgeRecord] = []
    metal_edge_records: list[ResidueMetalEdgeRecord] = []
    seen_residue_keys: set[tuple[int, int, str]] = set()
    seen_metal_keys: set[tuple[int, int, str]] = set()

    with ring_edges_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            interaction = normalize_ring_interaction_type(row.get("Interaction", ""))
            if interaction is None:
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
                src_idx, dst_idx, src_coord, dst_coord = canonicalize_edge_pair(src_idx, dst_idx, src_coord, dst_coord)
                edge_key = (src_idx, dst_idx, interaction)
                if edge_key in seen_residue_keys:
                    continue
                seen_residue_keys.add(edge_key)
                edge_dist_raw, edge_seqsep, edge_same_chain, vector_raw = build_pair_edge_geometry(
                    pocket.residues[src_idx],
                    pocket.residues[dst_idx],
                    src_coord=src_coord,
                    dst_coord=dst_coord,
                )
                residue_edge_records.append(
                    _build_residue_edge_record(
                        src_idx=src_idx,
                        dst_idx=dst_idx,
                        dist_raw=edge_dist_raw,
                        seqsep=edge_seqsep,
                        same_chain=edge_same_chain,
                        vector_raw=vector_raw,
                        interaction=interaction,
                        geometry_label="ring_atoms",
                    )
                )
                continue

            if interaction != "METAL_ION:SC_LIG":
                continue

            if src_is_residue and dst_is_metal:
                residue_idx = residue_to_index[src_key]
                residue = pocket.residues[residue_idx]
                residue_coord = resolve_ring_endpoint_coord(residue, row.get("Atom1", ""))
                metal_coord = metal_site_coord_map.get(dst_key)
                metal_idx = _resolve_metal_index(pocket, dst_key)
                if metal_coord is None or residue_coord is None or metal_idx is None:
                    continue
                edge_key = (residue_idx, metal_idx, interaction)
                if edge_key in seen_metal_keys:
                    continue
                seen_metal_keys.add(edge_key)
                metal_edge_records.append(
                    _build_residue_metal_edge_record(
                        residue_idx=residue_idx,
                        metal_idx=metal_idx,
                        residue_coord=residue_coord,
                        metal_coord=metal_coord,
                        interaction=interaction,
                    )
                )
                continue

            if dst_is_residue and src_is_metal:
                residue_idx = residue_to_index[dst_key]
                residue = pocket.residues[residue_idx]
                residue_coord = resolve_ring_endpoint_coord(residue, row.get("Atom2", ""))
                metal_coord = metal_site_coord_map.get(src_key)
                metal_idx = _resolve_metal_index(pocket, src_key)
                if metal_coord is None or residue_coord is None or metal_idx is None:
                    continue
                edge_key = (residue_idx, metal_idx, interaction)
                if edge_key in seen_metal_keys:
                    continue
                seen_metal_keys.add(edge_key)
                metal_edge_records.append(
                    _build_residue_metal_edge_record(
                        residue_idx=residue_idx,
                        metal_idx=metal_idx,
                        residue_coord=residue_coord,
                        metal_coord=metal_coord,
                        interaction=interaction,
                    )
                )

    return residue_edge_records, metal_edge_records
