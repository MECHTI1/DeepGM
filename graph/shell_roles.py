from __future__ import annotations

import torch
from torch import Tensor

from data_structures import DEFAULT_FIRST_SHELL_CUTOFF, PocketRecord
from featurization import MultinuclearSiteHandler, donor_coords_and_mask, functional_group_centroid, safe_norm


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
