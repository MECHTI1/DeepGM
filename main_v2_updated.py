#!/usr/bin/env python3
# TODO: Reconsider graph edge definition beyond only Cα radius, when will be added RING edges, they will may be inserted.
# TODO: One caution about angles
# TODO: The two-task loss is too simplistic.  currently do: loss = CE(metal) + CE(ec) -> FIX
# TODO: Need to download EC number of enzymes
# For all external features, like omega, normalize it (by using z-score).

"""
Stage-1 residue-level Zn-pocket classifier (late-fusion ESM + simplified GVP)
===========================================================================

What this file gives you
------------------------
1) A practical preprocessing scaffold for building one PyG graph per Zn-centered pocket
2) Residue-level node features split into two branches:
   - handcrafted / geometric node features that go into the GVP branch
   - precomputed ESM embeddings that are pooled separately and fused late
3) Simple geometric edges:
   - radius graph on CA coordinates (add edge when CA distance <= edge_radius)
   - scalar edge features = CA-CA and FG-FG distances
   - vector edge feature = normalized relative direction
4) A compact "GVP-like" model (simplified implementation, not exact paper code):
   - scalar channels + vector channels
   - message passing over the residue graph using only handcrafted/geometric features
   - separate ESM graph encoder
   - late fusion of graph-level GVP + ESM embeddings
   - dual heads for metal class and EC top-level class
5) Minimal training / evaluation utilities
6) A synthetic smoke test so you can verify shapes immediately

Important honesty note
----------------------
- This is a working scaffold and a GVP-style architecture, not a verbatim reimplementation
  of the original published GVP code.
- It is designed to be easy to read, modify, and debug first.
- Later, you can replace the SimpleGVP / SimpleGVPLayer blocks with a stricter GVP implementation
  without changing the dataset schema.

Expected environment
--------------------
Python >= 3.10
torch
torch_geometric
biopython (optional, only if you want to parse PDB/mmCIF directly)

Install example:
    pip install torch torch-geometric biopython
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import Dataset

try:
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
    from torch_geometric.nn import global_mean_pool, global_max_pool
except Exception as e:
    raise ImportError(
        "This script requires torch_geometric. Install it before running."
    ) from e
from Bio.PDB import MMCIFParser, PDBParser



# ============================================================
# Part 1: Constants and feature lists
# ============================================================

AA_ORDER = [
    "ALA", "ARG", "ASN", "ASP", "CYS",
    "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO",
    "SER", "THR", "TRP", "TYR", "VAL",
]
AA_TO_INDEX = {aa: i for i, aa in enumerate(AA_ORDER)}

NEGATIVE = {"ASP", "GLU"}
POSITIVE = {"ARG", "LYS", "HIS"}
POLAR = {"ASN", "ASP", "GLN", "GLU", "HIS", "LYS", "ARG", "SER", "THR", "TYR", "CYS"}
AROMATIC = {"PHE", "TYR", "TRP", "HIS"}
SULFUR = {"CYS", "MET"}
DONOR_CAPABLE = {"ARG", "ASN", "GLN", "HIS", "LYS", "SER", "THR", "TYR", "TRP", "CYS"}
ACCEPTOR_CAPABLE = {"ASP", "GLU", "ASN", "GLN", "HIS", "SER", "THR", "TYR", "CYS"}

BACKBONE_ATOMS = {"N", "CA", "C", "O", "OXT"}

DONOR_ATOMS_BY_RESIDUE = {
    "ASP": ["OD1", "OD2"],
    "GLU": ["OE1", "OE2"],
    "HIS": ["ND1", "NE2"],
    "CYS": ["SG"],
    "TYR": ["OH"],
    "SER": ["OG"],
    "THR": ["OG1"],
    "LYS": ["NZ"],
    "ARG": ["NH1", "NH2"],
    "ASN": ["OD1", "ND2"],
    "GLN": ["OE1", "NE2"],
    "TRP": ["NE1"],
}

DEFAULT_FIRST_SHELL_CUTOFF = 3.0
DEFAULT_SECOND_SHELL_CUTOFF = 4.5
DEFAULT_POCKET_RADIUS = 8.0
DEFAULT_EDGE_RADIUS = 6.0

# ------------------------------------------------------------------
# Recommended feature lists based on the canvas discussion.
# Important: many of these are NOT yet attached automatically in this
# script. Some are already handled; others are planned external features.
# ------------------------------------------------------------------
NODE_FEATURES_CURRENTLY_HANDLED = [
    # x_reschem
    "aa_one_hot",
    "charge_class_3way",
    "donor_flag",
    "acceptor_flag",
    "aromatic_flag",
    "sulfur_flag",
    "acidic_flag",
    "basic_flag",
    "histidine_flag",
    # x_role
    "is_first_shell",
    "is_second_shell",
    # x_dist_raw
    # "ca_to_metal",
    "fg_to_metal",
    # "min_donor_to_metal",
    # "second_min_donor_to_metal",
    # x_misc
    # "plddt",
    # x_env,
    "SASA",
    "BSA",
    "SolvEnergy",
    "fa_sol",
    "fa_elec",
    "pKa_shift",
    "dpKa_desolv",
    "dpKa_bg",
    "dpKa_titr",
    "omega",
    "rama_prepro",
    "fa_dun",
    # "fa_atr",
    # "fa_rep",
    # x_vec
    "v_fg_to_ca", #fg is functionla group
    "v_net_ligand",
    "v_res_to_metal",
    "cos_theta_bewteen_vnetligand_to_vrestometal",
]

# Recommended first-pass grouping for the structural branch.
# Important distinction:
#   - Burial / exposure is mostly about solvent access and desolvation.
#   - Packing / interaction terms are about residue-residue physical interactions.
# These are related, but they are not the same signal and should not be mentally merged.
NODE_FEATURE_GROUPS_RECOMMENDED = {
    "chemistry_identity": [
        "aa_one_hot",
        "charge_class_3way",
        "donor_flag",
        "acceptor_flag",
        "aromatic_flag",
        "sulfur_flag",
    ],
    "metal_geometry": [
        "is_first_shell",
        "is_second_shell",
        # "ca_to_metal",
        "fg_to_metal",
        # "min_donor_to_metal",
        "second_min_donor_to_metal",
    ],
    "burial_exposure": [
        "SASA",
        "BSA",
        "SolvEnergy",
        "fa_sol",
    ],
    # "packing_interactions": [
    #     "fa_atr",
    #     "fa_rep",
    #     "fa_elec",
    # ],
    "electrostatic_tuning": [
        "pKa_shift",
        "dpKa_desolv",
        "dpKa_bg",
        "dpKa_titr",
    ],
    "conformational_strain": [
        "omega",
        "rama_prepro",
        "fa_dun",
    ],
    # "confidence_and_pocket_position": [
        # "plddt",
    # ],
}

NODE_FEATURES_RECOMMENDED_KEEP_FIRST = [
    *NODE_FEATURE_GROUPS_RECOMMENDED["chemistry_identity"],
    *NODE_FEATURE_GROUPS_RECOMMENDED["metal_geometry"],
    *NODE_FEATURE_GROUPS_RECOMMENDED["burial_exposure"],
    *NODE_FEATURE_GROUPS_RECOMMENDED["packing_interactions"],
    *NODE_FEATURE_GROUPS_RECOMMENDED["electrostatic_tuning"],
    *NODE_FEATURE_GROUPS_RECOMMENDED["conformational_strain"],
    *NODE_FEATURE_GROUPS_RECOMMENDED["confidence_and_pocket_position"],
]

# NODE_FEATURES_OPTIONAL_LATER = [
#     "SolvExp",
#     "GBR6",
#     "mu2",
#     "mu3",
#     "mu4",
#     "ResSigDev",
#     "DestabRank",
#     "StabRank",
#     "lk_ball",
#     "lk_ball_iso",
#     "lk_ball_bridge",
#     "lk_ball_bridge_uncpl",
#     "total",
#     "fa_intra_atr_xover4",
#     "fa_intra_rep_xover4",
#     "fa_intra_sol_xover4",
#     "fa_intra_elec",
#     "fa_dun_dev",
#     "fa_dun_rot",
#     "fa_dun_semi",
#     "p_aa_pp",
#     "dslf_fa13",
#     "pro_close",
# ]

EDGE_FEATURES_RECOMMENDED_RING = [
    "ring_contact_type_one_hot",
    "contact_distance",
    "sequence_separation",
    "same_chain_flag",
]

HBOND_NODE_SUMMARIES_OPTIONAL_WITH_RING = [
    "hbond_sr_bb",
    "hbond_lr_bb",
    "hbond_bb_sc",
    "hbond_sc",
]


# ============================================================
# Part 2: Basic tensor / geometry helper functions
# ============================================================

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


# ============================================================
# Part 3: Dataclasses
# ============================================================

@dataclass
class ResidueRecord:
    chain_id: str
    resseq: int
    icode: str
    resname: str
    atoms: Dict[str, Tensor]
    plddt: float = 100.0

    esm_embedding: Optional[Tensor] = None
    is_first_shell: bool = False
    is_second_shell: bool = False

    # Optional external per-residue features (Rosetta / electrostatics / burial)
    external_features: Dict[str, float] = field(default_factory=dict)

    def residue_id(self) -> Tuple[str, int, str]:
        return (self.chain_id, self.resseq, self.icode)

    def get_atom(self, name: str) -> Optional[Tensor]:
        return self.atoms.get(name)

    def ca(self) -> Optional[Tensor]:
        return self.get_atom("CA")

    def get_external_feature(self, name: str, default: float = 0.0) -> float:
        return float(self.external_features.get(name, default))


@dataclass
class PocketRecord:
    structure_id: str
    pocket_id: str
    metal_element: str
    metal_coord: Tensor
    residues: List[ResidueRecord]
    y_metal: Optional[int] = None
    y_ec: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================
# Part 4: Residue chemistry and centroid logic
# ============================================================

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
        float(resname in SULFUR),
        float(resname in NEGATIVE),
        float(resname in POSITIVE),
        float(resname == "HIS"),
    ]
    return torch.tensor(flags, dtype=torch.float32)


def build_x_reschem(residue: ResidueRecord) -> Tensor:
    # 20 aa one-hot + 3 charge-class + 7 lean chemistry flags = 30 dims
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


# ============================================================
# Part 5: Optional structure parsing from PDB / mmCIF
# ============================================================

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
    plddt_values = []

    for atom in residue.get_atoms():
        name = atom.get_name().strip()
        coord = torch.tensor(atom.coord, dtype=torch.float32)
        atoms[name] = coord
        try:
            plddt_values.append(float(atom.bfactor))
        except Exception:
            pass

    if "CA" not in atoms:
        return None

    mean_plddt = float(sum(plddt_values) / max(1, len(plddt_values))) if plddt_values else 100.0
    parent_chain = residue.get_parent().id

    return ResidueRecord(
        chain_id=str(parent_chain),
        resseq=int(resseq),
        icode=str(icode).strip() if str(icode).strip() else "",
        resname=residue.resname.strip(),
        atoms=atoms,
        plddt=mean_plddt,
    )


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

        if len(pocket_residues) == 0:
            continue

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


# ============================================================
# Part 6: External feature attachment
# ============================================================

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
    """
    Attach external per-residue scalar features such as:
    burial/exposure: SASA, BSA, SolvEnergy, fa_sol
    packing/interactions: fa_atr, fa_rep, fa_elec
    electrostatic tuning: pKa_shift, dpKa_desolv, dpKa_bg, dpKa_titr
    conformational strain: omega, rama_prepro, fa_dun

    Keys are expected to match residue_id(): (chain_id, resseq, icode)
    """
    for rr in pocket.residues:
        key = rr.residue_id()
        if key in feature_lookup:
            rr.external_features.update(feature_lookup[key])
        elif strict:
            raise KeyError(f"Missing external feature dict for residue key {key}")


# ============================================================
# Part 7: Shell annotation
# ============================================================

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
        rr.is_first_shell = (min_d <= first_shell_cutoff)
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


# ============================================================
# Part 8: Graph construction
# ============================================================

def build_radius_graph(pos: Tensor, radius: float) -> Tensor:
    N = pos.size(0)
    dmat = pairwise_distances(pos)
    mask = (dmat <= radius) & (dmat > 0.0)

    src_list = []
    dst_list = []
    for i in range(N):
        js = torch.where(mask[i])[0]
        if js.numel() == 0:
            continue
        src_list.append(torch.full((js.numel(),), i, dtype=torch.long))
        dst_list.append(js.long())

    if len(src_list) == 0:
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


def build_external_feature_vector(rr: ResidueRecord) -> Tensor:
    """
    Core external features kept first from the canvas.
    Burial/exposure is kept separate conceptually from packing/interactions.
    Missing values default to 0.0.
    """
    names = [
        "SASA",
        "BSA",
        "SolvEnergy",
        "fa_sol",
        "fa_elec",
        "pKa_shift",
        "dpKa_desolv",
        "dpKa_bg",
        "dpKa_titr",
        "omega",
        "rama_prepro",
        "fa_dun",
        "fa_atr",
        "fa_rep",
    ]
    return torch.tensor([rr.get_external_feature(name, 0.0) for name in names], dtype=torch.float32)


def compute_net_ligand_vector(
    pocket: PocketRecord,
    ligand_cutoff: float = DEFAULT_FIRST_SHELL_CUTOFF,
    max_donors_per_residue: int = 2,
) -> Tensor:
    """
    Approximate pocket-level net ligand vector from donor atoms close to the metal.
    v_net = Σ_i (r_ligand_i - r_metal)
    """
    metal = pocket.metal_coord.float()
    v_net = torch.zeros(3, dtype=torch.float32)

    for rr in pocket.residues:
        donor_coords, donor_mask = donor_coords_and_mask(rr, max_donors=max_donors_per_residue)
        if not donor_mask.any():
            continue

        coords = donor_coords[donor_mask]
        d = safe_norm(coords - metal.unsqueeze(0), dim=-1)
        keep = d <= ligand_cutoff
        if keep.any():
            v_net = v_net + (coords[keep] - metal.unsqueeze(0)).sum(dim=0)

    return v_net


def residue_to_stage1_node_features(rr: ResidueRecord, metal_coord: Tensor, esm_dim: int, v_net: Tensor) -> Dict[str, Tensor]:
    if rr.esm_embedding is None:
        rr.esm_embedding = torch.zeros(esm_dim, dtype=torch.float32)

    ca = rr.ca()
    if ca is None:
        raise ValueError(f"Residue {rr.residue_id()} is missing CA coordinate.")

    fg = functional_group_centroid(rr)
    donor_coords, donor_mask = donor_coords_and_mask(rr, max_donors=2)

    ca_to_metal = float(safe_norm(ca - metal_coord, dim=-1).item())
    fg_to_metal = float(safe_norm(fg - metal_coord, dim=-1).item())
    min_donor_to_metal = min_distance_to_point(donor_coords, metal_coord, donor_mask)
    second_min_donor_to_metal = second_min_distance_to_point(donor_coords, metal_coord, donor_mask)

    x_role = torch.tensor(
        [float(rr.is_first_shell), float(rr.is_second_shell)],
        dtype=torch.float32,
    )

    x_dist_raw = torch.tensor(
        [ca_to_metal, fg_to_metal, min_donor_to_metal, second_min_donor_to_metal],
        dtype=torch.float32,
    )

    x_misc = torch.tensor(
        [float(rr.plddt)],
        dtype=torch.float32,
    )

    x_env = build_external_feature_vector(rr)

    v_res = (ca - metal_coord).float()
    v_net = v_net.float()
    v_net_norm = float(safe_norm(v_net, dim=-1).item())
    v_res_norm = float(safe_norm(v_res, dim=-1).item())
    denom = v_net_norm * v_res_norm + 1e-8
    cos_theta = float(torch.clamp(torch.dot(v_net, v_res) / denom, min=-1.0, max=1.0).item())

    x_misc = torch.cat(
        [
            x_misc,
            torch.tensor([v_net_norm, v_res_norm, cos_theta], dtype=torch.float32),
        ],
        dim=-1,
    )

    x_vec = torch.stack([(fg - ca).float(), v_net, v_res], dim=0)

    return {
        "x_esm": rr.esm_embedding.float(),
        "x_reschem": build_x_reschem(rr).float(),
        "x_role": x_role,
        "x_dist_raw": x_dist_raw,
        "x_misc": x_misc,
        "x_env": x_env,
        "x_vec": x_vec,
        "donor_coords": donor_coords.float(),
        "donor_mask": donor_mask,
        "fg_centroid": fg.float(),
        "pos": ca.float(),
    }


def pocket_to_pyg_data(
    pocket: PocketRecord,
    esm_dim: int,
    edge_radius: float = DEFAULT_EDGE_RADIUS,
) -> Data:
    annotate_shell_roles(pocket)
    v_net = compute_net_ligand_vector(pocket)

    node_dicts = [residue_to_stage1_node_features(rr, pocket.metal_coord, esm_dim, v_net) for rr in pocket.residues]
    N = len(node_dicts)

    x_esm = torch.stack([d["x_esm"] for d in node_dicts], dim=0)
    x_reschem = torch.stack([d["x_reschem"] for d in node_dicts], dim=0)
    x_role = torch.stack([d["x_role"] for d in node_dicts], dim=0)
    x_dist_raw = torch.stack([d["x_dist_raw"] for d in node_dicts], dim=0)
    x_misc = torch.stack([d["x_misc"] for d in node_dicts], dim=0)
    x_env = torch.stack([d["x_env"] for d in node_dicts], dim=0)
    x_vec = torch.stack([d["x_vec"] for d in node_dicts], dim=0)
    donor_coords = torch.stack([d["donor_coords"] for d in node_dicts], dim=0)
    donor_mask = torch.stack([d["donor_mask"] for d in node_dicts], dim=0)
    fg_centroid = torch.stack([d["fg_centroid"] for d in node_dicts], dim=0)
    pos = torch.stack([d["pos"] for d in node_dicts], dim=0)

    edge_index = build_radius_graph(pos, edge_radius)

    if edge_index.size(1) == 0:
        raise ValueError(
            f"Pocket {pocket.pocket_id} produced a graph with no edges at edge_radius={edge_radius}. "
            "Increase the radius or inspect the pocket residues."
        )

    src, dst = edge_index
    ca_ca_dist = safe_norm(pos[dst] - pos[src], dim=-1).unsqueeze(-1)

    fg_fg = []
    seq_sep = []
    same_chain = []
    for i, j in zip(src.tolist(), dst.tolist()):
        fg_fg.append(min_fg_fg_distance(pocket.residues[i], pocket.residues[j]))
        seq_sep.append(abs(pocket.residues[i].resseq - pocket.residues[j].resseq))
        same_chain.append(float(pocket.residues[i].chain_id == pocket.residues[j].chain_id))

    fg_fg_dist = torch.tensor(fg_fg, dtype=torch.float32).unsqueeze(-1)
    edge_dist_raw = torch.cat([ca_ca_dist, fg_fg_dist], dim=-1)
    edge_seqsep = torch.tensor(seq_sep, dtype=torch.float32).unsqueeze(-1)
    edge_same_chain = torch.tensor(same_chain, dtype=torch.float32).unsqueeze(-1)

    y_metal = None if pocket.y_metal is None else torch.tensor([pocket.y_metal], dtype=torch.long)
    y_ec = None if pocket.y_ec is None else torch.tensor([pocket.y_ec], dtype=torch.long)

    data = Data(
        x_esm=x_esm,
        x_reschem=x_reschem,
        x_role=x_role,
        x_dist_raw=x_dist_raw,
        x_misc=x_misc,
        x_env=x_env,
        x_vec=x_vec,
        pos=pos,
        fg_centroid=fg_centroid,
        donor_coords=donor_coords,
        donor_mask=donor_mask,
        edge_index=edge_index,
        edge_dist_raw=edge_dist_raw,
        edge_seqsep=edge_seqsep,
        edge_same_chain=edge_same_chain,
        zinc_pos=pocket.metal_coord.unsqueeze(0),
    )

    if y_metal is not None:
        data.y_metal = y_metal
    if y_ec is not None:
        data.y_ec = y_ec

    return data


# ============================================================
# Part 9: Dataset wrapper
# ============================================================

class PocketGraphDataset(Dataset):
    def __init__(
        self,
        pockets: List[PocketRecord],
        esm_dim: int,
        edge_radius: float = DEFAULT_EDGE_RADIUS,
    ):
        self.pockets = pockets
        self.esm_dim = esm_dim
        self.edge_radius = edge_radius

    def __len__(self) -> int:
        return len(self.pockets)

    def __getitem__(self, idx: int) -> Data:
        return pocket_to_pyg_data(
            self.pockets[idx],
            esm_dim=self.esm_dim,
            edge_radius=self.edge_radius,
        )


# ============================================================
# Part 10: RBF expansion
# ============================================================

class RBFExpansion(nn.Module):
    def __init__(self, n_rbf: int = 16, d_min: float = 0.0, d_max: float = 12.0):
        super().__init__()
        centers = torch.linspace(d_min, d_max, n_rbf)
        self.register_buffer("centers", centers)
        width = (d_max - d_min) / n_rbf
        self.gamma = 1.0 / (width * width + 1e-8)

    def forward(self, d: Tensor) -> Tensor:
        return torch.exp(-self.gamma * (d.unsqueeze(-1) - self.centers) ** 2)


# ============================================================
# Part 11: Encoders
# ============================================================

class NodeScalarEncoder(nn.Module):
    """
    Encode ONLY handcrafted / structural node scalar inputs into hidden scalar channels.

    Inputs:
      - x_reschem  [N, 30]
      - x_role     [N, 2]
      - x_dist_raw [N, 4]
      - x_misc     [N, 4] (plddt, ||v_net||, ||v_res||, cos(theta))
      - x_env      [N, 14]

    Distances are RBF-expanded internally.
    ESM is intentionally excluded here for late fusion.
    """
    def __init__(self, n_rbf: int = 16, out_dim: int = 128):
        super().__init__()
        self.dist_rbf = RBFExpansion(n_rbf=n_rbf, d_min=0.0, d_max=12.0)

        # 30 (reschem) + 2 (role) + 4 (misc) + 14 (external env) + 4*n_rbf
        in_dim = 30 + 2 + 4 + 14 + 4 * n_rbf

        self.out_proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.SiLU(),
        )

    def forward(self, x_reschem: Tensor, x_role: Tensor, x_dist_raw: Tensor, x_misc: Tensor, x_env: Tensor) -> Tensor:
        d_rbf = self.dist_rbf(x_dist_raw).flatten(start_dim=1)
        x = torch.cat([x_reschem, x_role, x_misc, x_env, d_rbf], dim=-1)
        return self.out_proj(x)


class ESMGraphEncoder(nn.Module):
    """
    Encode per-residue ESM embeddings separately and pool them at graph level.
    This branch is fused only after the GVP graph embedding is formed.
    """
    def __init__(self, esm_dim: int, proj_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.esm_proj = nn.Sequential(
            nn.Linear(esm_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x_esm: Tensor, batch: Tensor) -> Tensor:
        z = self.esm_proj(x_esm)
        z_mean = global_mean_pool(z, batch)
        z_max = global_max_pool(z, batch)
        return torch.cat([z_mean, z_max], dim=-1)


class EdgeScalarEncoder(nn.Module):
    """
    Encode raw edge scalars.

    Current inputs:
      - edge_dist_raw  [E, 2]  (CA-CA distance, FG-FG distance)
      - edge_seqsep    [E, 1]
      - edge_same_chain[E, 1]
    """
    def __init__(self, n_rbf: int = 16, out_dim: int = 64):
        super().__init__()
        self.dist_rbf = RBFExpansion(n_rbf=n_rbf, d_min=0.0, d_max=12.0)
        # 2 distance channels -> 2*n_rbf, plus seq sep and same-chain raw scalars
        in_dim = 2 * n_rbf + 2
        self.out_proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.SiLU(),
        )

    def forward(self, edge_dist_raw: Tensor, edge_seqsep: Tensor, edge_same_chain: Tensor) -> Tensor:
        d_rbf = self.dist_rbf(edge_dist_raw).flatten(start_dim=1)
        x = torch.cat([d_rbf, edge_seqsep, edge_same_chain], dim=-1)
        return self.out_proj(x)


# ============================================================
# Part 12: Simplified GVP-style blocks
# ============================================================

def vector_norm(v: Tensor, eps: float = 1e-8) -> Tensor:
    return torch.sqrt(torch.clamp((v * v).sum(dim=-1), min=eps))


class SimpleGVP(nn.Module):
    def __init__(self, s_in: int, v_in: int, s_out: int, v_out: int):
        super().__init__()
        self.scalar_mlp = nn.Sequential(
            nn.Linear(s_in + v_in, s_out),
            nn.SiLU(),
            nn.Linear(s_out, s_out),
        )
        self.vector_linear = nn.Linear(v_in, v_out, bias=False)
        self.vector_gate = nn.Linear(s_out, v_out)

    def forward(self, s: Tensor, v: Tensor) -> Tuple[Tensor, Tensor]:
        v_norm = vector_norm(v)
        s_cat = torch.cat([s, v_norm], dim=-1)
        s_out = self.scalar_mlp(s_cat)

        v_t = v.transpose(1, 2)
        v_proj = self.vector_linear(v_t).transpose(1, 2)

        gate = torch.sigmoid(self.vector_gate(s_out)).unsqueeze(-1)
        v_out = v_proj * gate

        return s_out, v_out


class SimpleGVPLayer(nn.Module):
    def __init__(self, s_dim: int, v_dim: int, e_dim: int):
        super().__init__()

        self.message_gvp = SimpleGVP(
            s_in=2 * s_dim + e_dim + 1,
            v_in=2 * v_dim + 1,
            s_out=s_dim,
            v_out=v_dim,
        )

        self.update_gvp = SimpleGVP(
            s_in=2 * s_dim,
            v_in=2 * v_dim,
            s_out=s_dim,
            v_out=v_dim,
        )

        self.norm_s = nn.LayerNorm(s_dim)

    def forward(self, s: Tensor, v: Tensor, edge_index: Tensor, edge_s: Tensor, edge_v: Tensor) -> Tuple[Tensor, Tensor]:
        src, dst = edge_index

        s_src = s[src]
        s_dst = s[dst]
        v_src = v[src]
        v_dst = v[dst]

        edge_len = vector_norm(edge_v)
        m_s_in = torch.cat([s_src, s_dst, edge_s, edge_len], dim=-1)
        m_v_in = torch.cat([v_src, v_dst, edge_v], dim=1)

        m_s, m_v = self.message_gvp(m_s_in, m_v_in)

        agg_s = torch.zeros_like(s)
        agg_s.index_add_(0, dst, m_s)

        agg_v = torch.zeros_like(v)
        agg_v.index_add_(0, dst, m_v)

        u_s_in = torch.cat([s, agg_s], dim=-1)
        u_v_in = torch.cat([v, agg_v], dim=1)
        ds, dv = self.update_gvp(u_s_in, u_v_in)

        s_out = self.norm_s(s + ds)
        v_out = v + dv
        return s_out, v_out


# ============================================================
# Part 13: Full graph classifier (late fusion)
# ============================================================

class GVPPocketClassifier(nn.Module):
    """
    Late-fusion design:
      - GVP branch sees only handcrafted/geometric node features
      - ESM branch is encoded separately and pooled per graph
      - fusion happens after graph-level pooling
    """
    def __init__(
        self,
        esm_dim: int,
        hidden_s: int = 128,
        hidden_v: int = 16,
        edge_hidden: int = 64,
        n_layers: int = 4,
        n_metal: int = 8,
        n_ec: int = 7,
        esm_fusion_dim: int = 128,
    ):
        super().__init__()

        self.node_scalar_encoder = NodeScalarEncoder(
            n_rbf=16,
            out_dim=hidden_s,
        )

        self.esm_graph_encoder = ESMGraphEncoder(
            esm_dim=esm_dim,
            proj_dim=esm_fusion_dim,
            dropout=0.1,
        )

        self.edge_scalar_encoder = EdgeScalarEncoder(
            n_rbf=16,
            out_dim=edge_hidden,
        )

        # x_vec channels: [fg-ca, v_net, v_res]
        self.init_vec_proj = nn.Linear(3, hidden_v, bias=False)

        self.layers = nn.ModuleList([
            SimpleGVPLayer(
                s_dim=hidden_s,
                v_dim=hidden_v,
                e_dim=edge_hidden,
            )
            for _ in range(n_layers)
        ])

        gvp_graph_dim = 2 * hidden_s
        esm_graph_dim = 2 * esm_fusion_dim
        fused_dim = gvp_graph_dim + esm_graph_dim

        self.head_metal = nn.Sequential(
            nn.Linear(fused_dim, hidden_s),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_s, n_metal),
        )

        self.head_ec = nn.Sequential(
            nn.Linear(fused_dim, hidden_s),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_s, n_ec),
        )

        self.ce = nn.CrossEntropyLoss()

    def _init_vector_channels(self, x_vec: Tensor) -> Tensor:
        x_t = x_vec.transpose(1, 2)
        x_proj = self.init_vec_proj(x_t)
        return x_proj.transpose(1, 2)

    def _prepare_edge_vectors(self, edge_index: Tensor, pos: Tensor) -> Tensor:
        src, dst = edge_index
        rel = pos[dst] - pos[src]
        rel = normalize_vec(rel, dim=-1)
        return rel.unsqueeze(1)

    def forward(self, data: Data) -> Dict[str, Tensor]:
        # 1) GVP branch: no ESM here
        s = self.node_scalar_encoder(
            data.x_reschem,
            data.x_role,
            data.x_dist_raw,
            data.x_misc,
            data.x_env,
        )

        v = self._init_vector_channels(data.x_vec)

        edge_s = self.edge_scalar_encoder(
            data.edge_dist_raw,
            data.edge_seqsep,
            data.edge_same_chain,
        )
        edge_v = self._prepare_edge_vectors(data.edge_index, data.pos)

        for layer in self.layers:
            s, v = layer(s, v, data.edge_index, edge_s, edge_v)

        pooled_mean = global_mean_pool(s, data.batch)
        pooled_max = global_max_pool(s, data.batch)
        gvp_graph_embed = torch.cat([pooled_mean, pooled_max], dim=-1)

        # 2) ESM branch: late fusion
        esm_graph_embed = self.esm_graph_encoder(data.x_esm, data.batch)

        # 3) Late fusion
        pocket_embed = torch.cat([gvp_graph_embed, esm_graph_embed], dim=-1)

        logits_metal = self.head_metal(pocket_embed)
        logits_ec = self.head_ec(pocket_embed)

        outputs = {
            "logits_metal": logits_metal,
            "logits_ec": logits_ec,
            "embed": pocket_embed,
            "gvp_embed": gvp_graph_embed,
            "esm_embed": esm_graph_embed,
        }

        if hasattr(data, "y_metal") and hasattr(data, "y_ec"):
            loss = self.ce(logits_metal, data.y_metal) + self.ce(logits_ec, data.y_ec)
            outputs["loss"] = loss

        return outputs


# ============================================================
# Part 14: Training and evaluation utilities
# ============================================================

def train_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, device: str = "cpu") -> float:
    model.train()
    total = 0.0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        model_outputs = model(batch)
        loss = model_outputs["loss"]
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total += float(loss.item())

    return total / max(1, len(loader))


@torch.no_grad()
def predict_batch(model: nn.Module, loader: DataLoader, device: str = "cpu") -> Dict[str, Tensor]:
    model.eval()

    metal_logits_all = []
    ec_logits_all = []
    metal_y_all = []
    ec_y_all = []

    for batch in loader:
        batch = batch.to(device)
        model_outputs = model(batch)

        metal_logits_all.append(model_outputs["logits_metal"].cpu())
        ec_logits_all.append(model_outputs["logits_ec"].cpu())

        if hasattr(batch, "y_metal"):
            metal_y_all.append(batch.y_metal.cpu())
        if hasattr(batch, "y_ec"):
            ec_y_all.append(batch.y_ec.cpu())

    result = {
        "metal_logits": torch.cat(metal_logits_all, dim=0),
        "ec_logits": torch.cat(ec_logits_all, dim=0),
    }
    if len(metal_y_all) > 0:
        result["metal_y"] = torch.cat(metal_y_all, dim=0)
    if len(ec_y_all) > 0:
        result["ec_y"] = torch.cat(ec_y_all, dim=0)

    return result


@torch.no_grad()
def accuracy_from_logits(logits: Tensor, y: Tensor) -> float:
    pred = logits.argmax(dim=-1)
    return float((pred == y).float().mean().item())


# ============================================================
# Part 15: Synthetic smoke-test data generation
# ============================================================

def random_residue(resname: Optional[str] = None, center: Optional[Tensor] = None) -> ResidueRecord:
    if resname is None:
        resname = random.choice(AA_ORDER)
    if center is None:
        center = torch.randn(3) * 3.0

    atoms = {}
    atoms["CA"] = center + torch.randn(3) * 0.2
    atoms["N"] = center + torch.tensor([-1.2, 0.3, 0.0]) + torch.randn(3) * 0.1
    atoms["C"] = center + torch.tensor([1.2, -0.2, 0.0]) + torch.randn(3) * 0.1
    atoms["O"] = center + torch.tensor([1.8, -0.5, 0.1]) + torch.randn(3) * 0.1

    for atom_name in donor_atom_names(resname):
        atoms[atom_name] = center + torch.randn(3) * 0.6 + torch.tensor([0.5, 0.5, 0.5])

    if all(a in BACKBONE_ATOMS for a in atoms.keys()):
        atoms["CB"] = center + torch.tensor([0.3, 1.1, 0.2]) + torch.randn(3) * 0.1

    rr = ResidueRecord(
        chain_id="A",
        resseq=random.randint(1, 999),
        icode="",
        resname=resname,
        atoms=atoms,
        plddt=float(random.uniform(70.0, 100.0)),
    )

    # Synthetic external feature placeholders so shapes work immediately
    rr.external_features = {
        "SASA": random.uniform(0.0, 100.0),
        "BSA": random.uniform(0.0, 100.0),
        "SolvEnergy": random.uniform(-5.0, 5.0),
        "fa_sol": random.uniform(-3.0, 3.0),
        "fa_elec": random.uniform(-3.0, 3.0),
        "pKa_shift": random.uniform(-4.0, 4.0),
        "dpKa_desolv": random.uniform(-4.0, 4.0),
        "dpKa_bg": random.uniform(-4.0, 4.0),
        "dpKa_titr": random.uniform(-4.0, 4.0),
        "omega": random.uniform(-2.0, 2.0),
        "rama_prepro": random.uniform(-2.0, 2.0),
        "fa_dun": random.uniform(-2.0, 2.0),
        "fa_atr": random.uniform(-3.0, 0.0),
        "fa_rep": random.uniform(0.0, 3.0),
    }
    return rr


def synthetic_pocket(
    pocket_id: str,
    n_residues: int,
    esm_dim: int,
    n_metal_classes: int = 8,
    n_ec_classes: int = 7,
) -> PocketRecord:
    metal = torch.zeros(3, dtype=torch.float32)

    residues = []
    for _ in range(n_residues):
        center = torch.randn(3) * 3.0
        rr = random_residue(center=center)
        rr.esm_embedding = torch.randn(esm_dim)
        residues.append(rr)

    return PocketRecord(
        structure_id="synthetic",
        pocket_id=pocket_id,
        metal_element="ZN",
        metal_coord=metal,
        residues=residues,
        y_metal=random.randint(0, n_metal_classes - 1),
        y_ec=random.randint(0, n_ec_classes - 1),
    )


# ============================================================
# Part 16: Smoke test
# ============================================================

def run_smoke_test(device: str = "cpu") -> None:
    torch.manual_seed(0)
    random.seed(0)

    esm_dim = 256

    pockets = [
        synthetic_pocket("p0", n_residues=14, esm_dim=esm_dim),
        synthetic_pocket("p1", n_residues=18, esm_dim=esm_dim),
        synthetic_pocket("p2", n_residues=11, esm_dim=esm_dim),
        synthetic_pocket("p3", n_residues=20, esm_dim=esm_dim),
    ]

    dataset = PocketGraphDataset(pockets, esm_dim=esm_dim, edge_radius=10.0)
    loader = DataLoader(dataset, batch_size=2, shuffle=True)

    model = GVPPocketClassifier(
        esm_dim=esm_dim,
        hidden_s=128,
        hidden_v=16,
        edge_hidden=64,
        n_layers=4,
        n_metal=8,
        n_ec=7,
        esm_fusion_dim=128,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

    loss = train_epoch(model, loader, optimizer, device=device)
    print(f"Smoke-test train loss: {loss:.4f}")

    result = predict_batch(model, loader, device=device)
    print("Metal logits shape:", tuple(result["metal_logits"].shape))
    print("EC logits shape:", tuple(result["ec_logits"].shape))
    print("Metal acc (random synthetic):", accuracy_from_logits(result["metal_logits"], result["metal_y"]))
    print("EC acc (random synthetic):", accuracy_from_logits(result["ec_logits"], result["ec_y"]))
    print("Smoke test completed successfully.")


# ============================================================
# Part 17: JSON debug helper
# ============================================================

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
                "plddt": rr.plddt,
                "is_first_shell": rr.is_first_shell,
                "is_second_shell": rr.is_second_shell,
                "external_features": rr.external_features,
                "atom_names": sorted(list(rr.atoms.keys())),
            }
            for rr in pocket.residues
        ],
    }

    with open(outpath, "w") as f:
        json.dump(payload, f, indent=2)


# ============================================================
# Part 18: Main entrypoint
# ============================================================

if __name__ == "__main__":
    run_smoke_test(device="cpu")
