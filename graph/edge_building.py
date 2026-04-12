from __future__ import annotations

import csv
from typing import Any

import torch
from torch import Tensor

from data_structures import (
    DEFAULT_FIRST_SHELL_CUTOFF,
    EDGE_SOURCE_TO_INDEX,
    EDGE_SOURCE_TYPES,
    INTERACTION_SUMMARIES_OPTIONAL_WITH_RING,
    PocketRecord,
    RING_INTERACTION_TO_INDEX,
    RING_INTERACTION_TYPES,
    ResidueRecord,
)
from featurization import (
    MultinuclearSiteHandler,
    donor_coords_and_mask,
    functional_group_centroid,
    one_hot_index,
    safe_norm,
)
from graph.ring_edges import (
    parse_ring_node_id,
    resolve_ring_edges_path,
    resolve_ring_endpoint_coord,
    ring_edges_path_candidates,
)


def annotate_shell_roles(
    pocket: PocketRecord,
    first_shell_cutoff: float = DEFAULT_FIRST_SHELL_CUTOFF,
    second_shell_cutoff: float = 4.5,
) -> None:
    metal_coords = MultinuclearSiteHandler.metal_coords_for_pocket(pocket)
    fg_centroids: list[Tensor] = []

    for residue in pocket.residues:
        donor_coords, donor_mask = donor_coords_and_mask(residue, max_donors=2)
        min_donor_distance = MultinuclearSiteHandler.min_distance_to_metals(
            donor_coords,
            metal_coords,
            donor_mask,
        )
        residue.is_first_shell = min_donor_distance <= first_shell_cutoff
        residue.is_second_shell = False
        fg_centroids.append(functional_group_centroid(residue))

    first_shell_centroids = [
        fg for residue, fg in zip(pocket.residues, fg_centroids) if residue.is_first_shell
    ]
    for residue, fg in zip(pocket.residues, fg_centroids):
        if residue.is_first_shell or not first_shell_centroids:
            continue
        residue.is_second_shell = min(
            safe_norm(fg - first_shell_fg, dim=-1).item() for first_shell_fg in first_shell_centroids
        ) <= second_shell_cutoff


def residue_atom_coords(residue: ResidueRecord) -> Tensor:
    return torch.stack([coord.float() for coord in residue.atoms.values()], dim=0)


def closest_points_between_residues(
    src_residue: ResidueRecord,
    dst_residue: ResidueRecord,
) -> tuple[Tensor, Tensor, float]:
    src_coords = residue_atom_coords(src_residue)
    dst_coords = residue_atom_coords(dst_residue)

    distances = safe_norm(src_coords[:, None, :] - dst_coords[None, :, :], dim=-1)
    flat_idx = int(torch.argmin(distances).item())
    src_idx = flat_idx // distances.size(1)
    dst_idx = flat_idx % distances.size(1)
    return src_coords[src_idx], dst_coords[dst_idx], float(distances[src_idx, dst_idx].item())


def build_pair_edge_geometry(
    src_residue: ResidueRecord,
    dst_residue: ResidueRecord,
    src_coord: Tensor | None = None,
    dst_coord: Tensor | None = None,
) -> tuple[Tensor, float, float, Tensor]:
    src_ca = src_residue.ca()
    dst_ca = dst_residue.ca()
    if src_ca is None or dst_ca is None:
        raise ValueError(
            f"Missing CA atom for edge pair {src_residue.residue_id()} -> {dst_residue.residue_id()}"
        )

    if src_coord is None or dst_coord is None:
        src_coord, dst_coord, contact_distance = closest_points_between_residues(src_residue, dst_residue)
    else:
        src_coord = src_coord.float()
        dst_coord = dst_coord.float()
        contact_distance = float(safe_norm(dst_coord - src_coord, dim=-1).item())

    vector_raw = (dst_coord - src_coord).float()
    ca_ca_distance = float(safe_norm(dst_ca - src_ca, dim=-1).item())
    edge_dist_raw = torch.tensor([contact_distance, ca_ca_distance], dtype=torch.float32)
    edge_seqsep = float(abs(src_residue.resseq - dst_residue.resseq))
    edge_same_chain = float(src_residue.chain_id == dst_residue.chain_id)
    return edge_dist_raw, edge_seqsep, edge_same_chain, vector_raw


def build_radius_graph_from_residues(residues: list[ResidueRecord], radius: float) -> Tensor:
    edges: list[tuple[int, int]] = []
    for src_idx, src_residue in enumerate(residues):
        for dst_idx, dst_residue in enumerate(residues):
            if src_idx == dst_idx:
                continue
            _src_coord, _dst_coord, contact_distance = closest_points_between_residues(src_residue, dst_residue)
            if contact_distance <= radius:
                edges.append((src_idx, dst_idx))

    if not edges:
        return torch.zeros(2, 0, dtype=torch.long)
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


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
    edge_records: list[dict[str, Any]] = []

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
            if float(safe_norm((dst_coord - src_coord).float(), dim=-1).item()) <= 1e-8:
                continue

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

    return edge_records


def radius_edge_records_from_index(
    pocket: PocketRecord,
    base_edge_index: Tensor,
) -> list[dict[str, Any]]:
    edge_records: list[dict[str, Any]] = []
    if base_edge_index.size(1) == 0:
        return edge_records

    src_indices, dst_indices = base_edge_index
    for src_idx, dst_idx in zip(src_indices.tolist(), dst_indices.tolist()):
        src_residue = pocket.residues[src_idx]
        dst_residue = pocket.residues[dst_idx]
        edge_dist_raw, edge_seqsep, edge_same_chain, vector_raw = build_pair_edge_geometry(
            src_residue,
            dst_residue,
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


def stack_edge_features(edge_records: list[dict[str, Any]]) -> dict[str, Tensor]:
    return {
        "edge_index": torch.tensor(
            [[record["src"] for record in edge_records], [record["dst"] for record in edge_records]],
            dtype=torch.long,
        ),
        "edge_dist_raw": torch.stack([record["dist_raw"] for record in edge_records], dim=0),
        "edge_seqsep": torch.tensor([record["seqsep"] for record in edge_records], dtype=torch.float32).unsqueeze(-1),
        "edge_same_chain": torch.tensor(
            [record["same_chain"] for record in edge_records],
            dtype=torch.float32,
        ).unsqueeze(-1),
        "edge_vector_raw": torch.stack([record["vector_raw"] for record in edge_records], dim=0),
        "edge_interaction_type": torch.stack([record["interaction_type"] for record in edge_records], dim=0),
        "edge_source_type": torch.stack([record["source_type"] for record in edge_records], dim=0),
    }


__all__ = [
    "annotate_shell_roles",
    "build_pair_edge_geometry",
    "build_radius_graph_from_residues",
    "build_ring_interaction_edge_records",
    "closest_points_between_residues",
    "radius_edge_records_from_index",
    "residue_atom_coords",
    "stack_edge_features",
]
