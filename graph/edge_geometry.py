from __future__ import annotations

import torch
from torch import Tensor

from data_structures import ResidueRecord
from featurization import safe_norm


def residue_atom_coords(residue: ResidueRecord) -> Tensor:
    return torch.stack([coord.float() for coord in residue.atoms.values()], dim=0)


def residue_atom_coords_list(residues: list[ResidueRecord]) -> list[Tensor]:
    return [residue_atom_coords(residue) for residue in residues]


def residue_spatial_envelope(
    residue: ResidueRecord,
    coords: Tensor | None = None,
) -> tuple[Tensor, float]:
    if coords is None:
        coords = residue_atom_coords(residue)
    center = coords.mean(dim=0)
    radius = float(safe_norm(coords - center.unsqueeze(0), dim=-1).max().item())
    return center, radius


def candidate_residue_pairs_within_radius(
    residues: list[ResidueRecord],
    radius: float,
    atom_coords_by_residue: list[Tensor] | None = None,
) -> list[tuple[int, int]]:
    if len(residues) < 2:
        return []

    if atom_coords_by_residue is None:
        atom_coords_by_residue = residue_atom_coords_list(residues)

    envelopes = [
        residue_spatial_envelope(residue, coords=coords)
        for residue, coords in zip(residues, atom_coords_by_residue)
    ]
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


def closest_points_between_coord_tensors(
    src_coords: Tensor,
    dst_coords: Tensor,
) -> tuple[Tensor, Tensor, float]:
    distances = safe_norm(src_coords[:, None, :] - dst_coords[None, :, :], dim=-1)
    flat_idx = int(torch.argmin(distances).item())
    src_idx = flat_idx // distances.size(1)
    dst_idx = flat_idx % distances.size(1)
    return src_coords[src_idx], dst_coords[dst_idx], float(distances[src_idx, dst_idx].item())


def closest_points_between_residues(
    src_residue: ResidueRecord,
    dst_residue: ResidueRecord,
    src_coords: Tensor | None = None,
    dst_coords: Tensor | None = None,
) -> tuple[Tensor, Tensor, float]:
    if src_coords is None:
        src_coords = residue_atom_coords(src_residue)
    if dst_coords is None:
        dst_coords = residue_atom_coords(dst_residue)
    return closest_points_between_coord_tensors(src_coords, dst_coords)


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


def canonicalize_edge_pair(
    src_idx: int,
    dst_idx: int,
    src_coord: Tensor | None = None,
    dst_coord: Tensor | None = None,
) -> tuple[int, int, Tensor | None, Tensor | None]:
    if src_idx <= dst_idx:
        return src_idx, dst_idx, src_coord, dst_coord
    return dst_idx, src_idx, dst_coord, src_coord


def build_radius_graph_from_residues(residues: list[ResidueRecord], radius: float) -> Tensor:
    atom_coords_by_residue = residue_atom_coords_list(residues)
    edges: list[tuple[int, int]] = []
    for src_idx, dst_idx in candidate_residue_pairs_within_radius(
        residues,
        radius,
        atom_coords_by_residue=atom_coords_by_residue,
    ):
        _src_coord, _dst_coord, contact_distance = closest_points_between_residues(
            residues[src_idx],
            residues[dst_idx],
            src_coords=atom_coords_by_residue[src_idx],
            dst_coords=atom_coords_by_residue[dst_idx],
        )
        if contact_distance <= radius:
            edges.append((src_idx, dst_idx))

    if not edges:
        return torch.zeros(2, 0, dtype=torch.long)
    return torch.tensor(edges, dtype=torch.long).t().contiguous()
