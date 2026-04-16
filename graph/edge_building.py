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

RING_EDGE_SYMMETRY_POLICY = "bidirectional"
RING_METAL_CONTACT_POLICY = "self_loop"


def compute_shell_roles(
    pocket: PocketRecord,
    first_shell_cutoff: float = DEFAULT_FIRST_SHELL_CUTOFF,
    second_shell_cutoff: float = 4.5,
) -> list[tuple[bool, bool]]:
    metal_coords = MultinuclearSiteHandler.metal_coords_for_pocket(pocket)
    fg_centroids: list[Tensor] = []
    first_shell_flags: list[bool] = []

    for residue in pocket.residues:
        donor_coords, donor_mask = donor_coords_and_mask(residue, max_donors=2)
        min_donor_distance = MultinuclearSiteHandler.min_distance_to_metals(
            donor_coords,
            metal_coords,
            donor_mask,
        )
        first_shell_flags.append(min_donor_distance <= first_shell_cutoff)
        fg_centroids.append(functional_group_centroid(residue))

    first_shell_centroids = [
        fg for is_first_shell, fg in zip(first_shell_flags, fg_centroids) if is_first_shell
    ]
    second_shell_flags: list[bool] = []
    for is_first_shell, fg in zip(first_shell_flags, fg_centroids):
        if is_first_shell or not first_shell_centroids:
            second_shell_flags.append(False)
            continue
        second_shell_flags.append(
            min(safe_norm(fg - first_shell_fg, dim=-1).item() for first_shell_fg in first_shell_centroids)
            <= second_shell_cutoff
        )

    return list(zip(first_shell_flags, second_shell_flags))


def annotate_shell_roles(
    pocket: PocketRecord,
    first_shell_cutoff: float = DEFAULT_FIRST_SHELL_CUTOFF,
    second_shell_cutoff: float = 4.5,
) -> None:
    shell_roles = compute_shell_roles(
        pocket,
        first_shell_cutoff=first_shell_cutoff,
        second_shell_cutoff=second_shell_cutoff,
    )
    for residue, (is_first_shell, is_second_shell) in zip(pocket.residues, shell_roles):
        residue.is_first_shell = is_first_shell
        residue.is_second_shell = is_second_shell


def residue_atom_coords(residue: ResidueRecord) -> Tensor:
    return torch.stack([coord.float() for coord in residue.atoms.values()], dim=0)


def residue_spatial_envelope(residue: ResidueRecord) -> tuple[Tensor, float]:
    coords = residue_atom_coords(residue)
    center = coords.mean(dim=0)
    radius = float(safe_norm(coords - center.unsqueeze(0), dim=-1).max().item())
    return center, radius


def candidate_residue_pairs_within_radius(
    residues: list[ResidueRecord],
    radius: float,
) -> list[tuple[int, int]]:
    if len(residues) < 2:
        return []

    envelopes = [residue_spatial_envelope(residue) for residue in residues]
    centers = [center for center, _radius in envelopes]
    envelope_radii = [envelope_radius for _center, envelope_radius in envelopes]
    max_envelope_radius = max(envelope_radii, default=0.0)
    cell_size = max(radius + 2.0 * max_envelope_radius, 1e-6)

    buckets: dict[tuple[int, int, int], list[int]] = {}
    for idx, center in enumerate(centers):
        cell = tuple(torch.floor(center / cell_size).to(torch.long).tolist())
        buckets.setdefault(cell, []).append(idx)

    offsets = (-1, 0, 1)
    candidate_pairs: set[tuple[int, int]] = set()
    for src_idx, src_center in enumerate(centers):
        src_cell = tuple(torch.floor(src_center / cell_size).to(torch.long).tolist())
        for dx in offsets:
            for dy in offsets:
                for dz in offsets:
                    neighbor_cell = (src_cell[0] + dx, src_cell[1] + dy, src_cell[2] + dz)
                    for dst_idx in buckets.get(neighbor_cell, []):
                        if dst_idx <= src_idx:
                            continue
                        coarse_cutoff = radius + envelope_radii[src_idx] + envelope_radii[dst_idx]
                        center_distance = float(safe_norm(centers[dst_idx] - src_center, dim=-1).item())
                        if center_distance <= coarse_cutoff:
                            candidate_pairs.add((src_idx, dst_idx))

    return sorted(candidate_pairs)


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
    adjacency: dict[int, list[int]] = {idx: [] for idx in range(len(residues))}
    for src_idx, dst_idx in candidate_residue_pairs_within_radius(residues, radius):
        _src_coord, _dst_coord, contact_distance = closest_points_between_residues(
            residues[src_idx],
            residues[dst_idx],
        )
        if contact_distance <= radius:
            adjacency[src_idx].append(dst_idx)
            adjacency[dst_idx].append(src_idx)

    edges = [
        (src_idx, dst_idx)
        for src_idx in range(len(residues))
        for dst_idx in sorted(adjacency[src_idx])
    ]

    if not edges:
        return torch.zeros(2, 0, dtype=torch.long)
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def build_radius_edge_records_from_residues(
    pocket: PocketRecord,
    radius: float,
) -> list[dict[str, Any]]:
    edge_records: list[dict[str, Any]] = []
    adjacency: dict[int, list[int]] = {idx: [] for idx in range(len(pocket.residues))}
    directed_contacts: dict[tuple[int, int], tuple[Tensor, Tensor]] = {}
    for src_idx, dst_idx in candidate_residue_pairs_within_radius(pocket.residues, radius):
        src_residue = pocket.residues[src_idx]
        dst_residue = pocket.residues[dst_idx]
        src_coord, dst_coord, contact_distance = closest_points_between_residues(src_residue, dst_residue)
        if contact_distance > radius:
            continue
        directed_contacts[(src_idx, dst_idx)] = (src_coord, dst_coord)
        directed_contacts[(dst_idx, src_idx)] = (dst_coord, src_coord)
        adjacency[src_idx].append(dst_idx)
        adjacency[dst_idx].append(src_idx)

    for src_idx in range(len(pocket.residues)):
        for dst_idx in sorted(adjacency[src_idx]):
            src_residue = pocket.residues[src_idx]
            dst_residue = pocket.residues[dst_idx]
            src_coord, dst_coord = directed_contacts[(src_idx, dst_idx)]
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
        src_coord: Tensor,
        dst_coord: Tensor,
    ) -> None:
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
                if RING_EDGE_SYMMETRY_POLICY == "bidirectional" and src_idx != dst_idx:
                    append_edge_record(
                        src_idx=dst_idx,
                        dst_idx=src_idx,
                        interaction=interaction,
                        src_coord=dst_coord,
                        dst_coord=src_coord,
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
                if metal_coord is None:
                    continue
                if residue_coord is None:
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
                if metal_coord is None:
                    continue
                if residue_coord is None:
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
