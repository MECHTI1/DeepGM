from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor

from data_structures import (
    AA_ORDER,
    AA_TO_INDEX,
    ACCEPTOR_CAPABLE,
    AROMATIC,
    BACKBONE_ATOMS,
    DEFAULT_FIRST_SHELL_CUTOFF,
    DONOR_ATOMS_BY_RESIDUE,
    DONOR_CAPABLE,
    NEGATIVE,
    POSITIVE,
    PocketRecord,
    ResidueRecord,
)

BURIAL_FEATURE_NAMES = ("SASA", "BSA", "SolvEnergy", "fa_sol")
PKA_FEATURE_NAMES = ("pKa_shift", "dpKa_desolv", "dpKa_bg", "dpKa_titr")
CONFORMATION_FEATURE_NAMES = ("omega", "rama_prepro", "fa_dun")
INTERACTION_FEATURE_NAMES = ("fa_elec", "fa_atr", "fa_rep")


def safe_norm(x: Tensor, dim: int = -1, keepdim: bool = False, eps: float = 1e-8) -> Tensor:
    return torch.sqrt(torch.clamp((x * x).sum(dim=dim, keepdim=keepdim), min=eps))


def normalize_vec(x: Tensor, dim: int = -1, eps: float = 1e-8) -> Tensor:
    return x / safe_norm(x, dim=dim, keepdim=True, eps=eps)


def pairwise_distances(x: Tensor) -> Tensor:
    diff = x[:, None, :] - x[None, :, :]
    return safe_norm(diff, dim=-1)


def one_hot_index(index: int, size: int) -> Tensor:
    one_hot = torch.zeros(size, dtype=torch.float32)
    if 0 <= index < size:
        one_hot[index] = 1.0
    return one_hot


def residue_one_hot(resname: str) -> Tensor:
    idx = AA_TO_INDEX.get(resname, -1)
    return one_hot_index(idx, len(AA_ORDER))


def residue_charge_class(resname: str) -> Tensor:
    if resname in NEGATIVE:
        return torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
    if resname in POSITIVE:
        return torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
    return torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)


def residue_chemistry_flags(resname: str) -> Tensor:
    flags = [
        float(resname in DONOR_CAPABLE),
        float(resname in ACCEPTOR_CAPABLE),
        float(resname in AROMATIC),
        float(resname in NEGATIVE),
        float(resname in POSITIVE),
    ]
    return torch.tensor(flags, dtype=torch.float32)


def build_x_reschem(residue: ResidueRecord) -> Tensor:
    return torch.cat(
        [
            residue_one_hot(residue.resname),
            residue_charge_class(residue.resname),
            residue_chemistry_flags(residue.resname),
        ],
        dim=-1,
    )


def donor_atom_names(resname: str) -> List[str]:
    return DONOR_ATOMS_BY_RESIDUE.get(resname, [])[:2]


def donor_coords_and_mask(residue: ResidueRecord, max_donors: int = 2) -> Tuple[Tensor, Tensor]:
    coords = torch.zeros(max_donors, 3, dtype=torch.float32)
    mask = torch.zeros(max_donors, dtype=torch.bool)

    names = donor_atom_names(residue.resname)
    for i, atom_name in enumerate(names[:max_donors]):
        atom = residue.get_atom(atom_name)
        if atom is not None:
            coords[i] = atom.float()
            mask[i] = True

    return coords, mask


def sidechain_atoms(residue: ResidueRecord) -> List[Tensor]:
    sidechain = []
    for atom_name, coord in residue.atoms.items():
        if atom_name not in BACKBONE_ATOMS:
            sidechain.append(coord.float())
    return sidechain


def centroid(coords: List[Tensor]) -> Optional[Tensor]:
    if len(coords) == 0:
        return None
    return torch.stack(coords, dim=0).mean(dim=0)


def functional_group_centroid(residue: ResidueRecord) -> Tensor:
    donor_coords, donor_mask = donor_coords_and_mask(residue, max_donors=2)
    if donor_mask.any():
        return donor_coords[donor_mask].mean(dim=0)

    sc = sidechain_atoms(residue)
    sc_cent = centroid(sc)
    if sc_cent is not None:
        return sc_cent

    ca = residue.ca()
    if ca is None:
        raise ValueError(f"Residue {residue.residue_id()} has no CA and no usable centroid.")
    return ca.float()


def min_distance_to_point(coords: Tensor, point: Tensor, mask: Optional[Tensor] = None) -> float:
    if coords.numel() == 0:
        return 999.0
    if mask is not None:
        coords = coords[mask]
    if coords.numel() == 0:
        return 999.0
    return float(safe_norm(coords - point.unsqueeze(0), dim=-1).min().item())


def second_min_distance_to_point(coords: Tensor, point: Tensor, mask: Optional[Tensor] = None) -> float:
    if coords.numel() == 0:
        return 999.0
    if mask is not None:
        coords = coords[mask]
    if coords.numel() == 0:
        return 999.0
    d = safe_norm(coords - point.unsqueeze(0), dim=-1)
    vals, _ = torch.sort(d)
    if vals.numel() == 1:
        return float(vals[0].item())
    return float(vals[1].item())


def build_external_feature_vector(rr: ResidueRecord, feature_names: Tuple[str, ...]) -> Tensor:
    return torch.tensor(
        [rr.get_external_feature(name, 0.0) for name in feature_names],
        dtype=torch.float32,
    )


def build_external_feature_groups(rr: ResidueRecord) -> Dict[str, Tensor]:
    return {
        "burial": build_external_feature_vector(rr, BURIAL_FEATURE_NAMES),
        "pka": build_external_feature_vector(rr, PKA_FEATURE_NAMES),
        "conformation": build_external_feature_vector(rr, CONFORMATION_FEATURE_NAMES),
        "interactions": build_external_feature_vector(rr, INTERACTION_FEATURE_NAMES),
    }


class MultinuclearSiteHandler:
    @staticmethod
    def metal_coords_for_pocket(pocket: PocketRecord) -> Tensor:
        metal_coords = pocket.metal_coords
        return torch.stack([coord.float() for coord in metal_coords], dim=0)

    @staticmethod
    def nearest_metal_for_points(points: Tensor, metal_coords: Tensor) -> Tuple[Tensor, Tensor]:
        diff = points[:, None, :] - metal_coords[None, :, :]
        dists = safe_norm(diff, dim=-1)
        min_dists, metal_idx = torch.min(dists, dim=1)
        nearest = metal_coords[metal_idx]
        return nearest, min_dists

    @staticmethod
    def nearest_metal_for_point(point: Tensor, metal_coords: Tensor) -> Tuple[Tensor, float]:
        nearest, min_dists = MultinuclearSiteHandler.nearest_metal_for_points(
            point.unsqueeze(0),
            metal_coords,
        )
        return nearest[0], float(min_dists[0].item())

    @staticmethod
    def min_distance_to_metals(coords: Tensor, metal_coords: Tensor, mask: Optional[Tensor] = None) -> float:
        if coords.numel() == 0:
            return 999.0
        if mask is not None:
            coords = coords[mask]
        if coords.numel() == 0:
            return 999.0
        _, min_dists = MultinuclearSiteHandler.nearest_metal_for_points(coords, metal_coords)
        return float(min_dists.min().item())

    @staticmethod
    def site_metal_stats(pocket: PocketRecord) -> Tensor:
        # Pocket-level metal-site summary used later in late fusion.
        metal_coords = MultinuclearSiteHandler.metal_coords_for_pocket(pocket)
        metal_count = float(metal_coords.size(0))
        is_multinuclear = float(metal_count > 1.0)

        if metal_coords.size(0) <= 1:
            min_dist = 0.0
            mean_dist = 0.0
        else:
            dmat = pairwise_distances(metal_coords)
            mask = torch.triu(torch.ones_like(dmat, dtype=torch.bool), diagonal=1)
            pair_dists = dmat[mask]
            min_dist = float(pair_dists.min().item())
            mean_dist = float(pair_dists.mean().item())

        return torch.tensor(
            [is_multinuclear, metal_count, min_dist, mean_dist],
            dtype=torch.float32,
        )


def compute_net_ligand_vector(
    pocket: PocketRecord,
    ligand_cutoff: float = DEFAULT_FIRST_SHELL_CUTOFF,
    max_donors_per_residue: int = 2,
) -> Tensor:
    metal_coords = MultinuclearSiteHandler.metal_coords_for_pocket(pocket)
    v_net = torch.zeros(3, dtype=torch.float32)

    for rr in pocket.residues:
        donor_coords, donor_mask = donor_coords_and_mask(rr, max_donors=max_donors_per_residue)
        if not donor_mask.any():
            continue

        coords = donor_coords[donor_mask]
        nearest_metals, min_dists = MultinuclearSiteHandler.nearest_metal_for_points(coords, metal_coords)
        keep = min_dists <= ligand_cutoff
        if keep.any():
            # Sum ligand-to-metal directions over direct binders to get one site-level orientation vector.
            v_net = v_net + (coords[keep] - nearest_metals[keep]).sum(dim=0)

    return v_net


def residue_to_stage1_node_features(
    rr: ResidueRecord,
    pocket: PocketRecord,
    esm_dim: int,
    v_net: Tensor,
) -> Dict[str, Tensor]:
    if rr.esm_embedding is None:
        rr.esm_embedding = torch.zeros(esm_dim, dtype=torch.float32)

    metal_coords = MultinuclearSiteHandler.metal_coords_for_pocket(pocket)
    ca = rr.ca()
    fg = functional_group_centroid(rr)
    donor_coords, donor_mask = donor_coords_and_mask(rr, max_donors=2)

    nearest_metal_to_ca, ca_to_metal = MultinuclearSiteHandler.nearest_metal_for_point(ca.float(), metal_coords)
    _, fg_to_metal = MultinuclearSiteHandler.nearest_metal_for_point(fg.float(), metal_coords)
    min_donor_to_metal = MultinuclearSiteHandler.min_distance_to_metals(donor_coords, metal_coords, donor_mask)

    x_role = torch.tensor(
        [float(rr.is_first_shell), float(rr.is_second_shell)],
        dtype=torch.float32,
    )
    x_dist_raw = torch.tensor(
        [ca_to_metal, fg_to_metal, min_donor_to_metal],
        dtype=torch.float32,
    )

    # v_res anchors the residue to its nearest metal; x_misc stores norms and the angle proxy to v_net.
    v_res = (ca.float() - nearest_metal_to_ca).float()
    v_net = v_net.float()
    v_net_norm = float(safe_norm(v_net, dim=-1).item())
    v_res_norm = float(safe_norm(v_res, dim=-1).item())
    denom = v_net_norm * v_res_norm + 1e-8
    cos_theta = float(torch.clamp(torch.dot(v_net, v_res) / denom, min=-1.0, max=1.0).item())

    x_misc = torch.tensor([v_net_norm, v_res_norm, cos_theta], dtype=torch.float32)
    env_groups = build_external_feature_groups(rr)
    # Two node vector channels: local residue geometry and residue-to-metal direction.
    x_vec = torch.stack([(fg - ca).float(), v_res], dim=0)

    return {
        "x_esm": rr.esm_embedding.float(),
        "x_reschem": build_x_reschem(rr).float(),
        "x_role": x_role,
        "x_dist_raw": x_dist_raw,
        "x_misc": x_misc,
        "x_env_burial": env_groups["burial"],
        "x_env_pka": env_groups["pka"],
        "x_env_conf": env_groups["conformation"],
        "x_env_interactions": env_groups["interactions"],
        "x_vec": x_vec,
        "donor_coords": donor_coords.float(),
        "donor_mask": donor_mask,
        "fg_centroid": fg.float(),
        "pos": ca.float(),
    }
