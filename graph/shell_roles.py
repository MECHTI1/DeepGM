from __future__ import annotations

from data_structures import (
    DEFAULT_FIRST_SHELL_CUTOFF,
    PocketRecord,
)
from featurization import MultinuclearSiteHandler, donor_coords_and_mask, functional_group_centroid, safe_norm
from graph.edge_sources import build_ring_edge_records
from graph.ring_edges import resolve_ring_edges_path


def _compute_first_shell_flags(
    pocket: PocketRecord,
    *,
    first_shell_cutoff: float,
) -> list[bool]:
    metal_coords = MultinuclearSiteHandler.metal_coords_for_pocket(pocket)
    first_shell_flags: list[bool] = []
    for residue in pocket.residues:
        donor_coords, donor_mask = donor_coords_and_mask(residue, max_donors=2)
        min_donor_distance = MultinuclearSiteHandler.min_distance_to_metals(
            donor_coords,
            metal_coords,
            donor_mask,
        )
        first_shell_flags.append(min_donor_distance <= first_shell_cutoff)
    return first_shell_flags


def _compute_second_shell_flags_from_ring(
    pocket: PocketRecord,
    *,
    first_shell_flags: list[bool],
) -> list[bool] | None:
    if resolve_ring_edges_path(pocket) is None:
        return None

    residue_edge_records, _metal_edge_records = build_ring_edge_records(
        pocket,
        require_ring_edges=False,
    )
    second_shell_flags = [False] * len(pocket.residues)
    for edge_record in residue_edge_records:
        src_idx = int(edge_record.src)
        dst_idx = int(edge_record.dst)
        if src_idx == dst_idx:
            continue

        src_is_first_shell = first_shell_flags[src_idx]
        dst_is_first_shell = first_shell_flags[dst_idx]
        if src_is_first_shell and not dst_is_first_shell:
            second_shell_flags[dst_idx] = True
        elif dst_is_first_shell and not src_is_first_shell:
            second_shell_flags[src_idx] = True

    return second_shell_flags


def _compute_second_shell_flags_by_centroid(
    pocket: PocketRecord,
    *,
    first_shell_flags: list[bool],
    second_shell_cutoff: float,
) -> list[bool]:
    fg_centroids = [functional_group_centroid(residue) for residue in pocket.residues]
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
    return second_shell_flags


def compute_shell_roles(
    pocket: PocketRecord,
    first_shell_cutoff: float = DEFAULT_FIRST_SHELL_CUTOFF,
    second_shell_cutoff: float = 4.5,
) -> list[tuple[bool, bool]]:
    first_shell_flags = _compute_first_shell_flags(
        pocket,
        first_shell_cutoff=first_shell_cutoff,
    )
    second_shell_flags = _compute_second_shell_flags_from_ring(
        pocket,
        first_shell_flags=first_shell_flags,
    )
    if second_shell_flags is None:
        second_shell_flags = _compute_second_shell_flags_by_centroid(
            pocket,
            first_shell_flags=first_shell_flags,
            second_shell_cutoff=second_shell_cutoff,
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
