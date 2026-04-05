#!/usr/bin/env python3
"""
Stage-1 residue-level Zn-pocket classifier (GVP-first, simplified)
==================================================================

What this file gives you
------------------------
1) A practical preprocessing scaffold for building one PyG graph per Zn-centered pocket
2) Residue-level node features:
   - ESM embedding (precomputed outside this script, then attached by key)
   - residue chemistry flags
   - shell role flags
   - metal-distance features
   - wall / pLDDT placeholders (kept simple in stage-1; replaceable later)
3) Simple geometric edges:
   - radius graph on CA coordinates (add edge when CA distance <= edge_radius)
   - scalar edge features = CA-CA and FG-FG distances (simple, stable stage-1 edge geometry)
   - vector edge feature = normalized relative direction
4) A compact "GVP-like" model (simplified implementation, not exact paper code):
   - scalar channels + vector channels
   - message passing over the residue graph
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

# ============================================================
# Part 1: Imports and global constants
# This section collects all imports and residue-level chemistry
# constants in one place, so the rest of the code stays clean.
# ============================================================
# Example input (Part 1):
#   1) Import this file in Python.
#   2) Query an amino-acid code like "HIS".
# Example output (Part 1):
#   1) Global constants are initialized.
#   2) Example mapping: AA_TO_INDEX["HIS"] == 8.

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

try:
    from Bio.PDB import MMCIFParser, PDBParser
    BIOPYTHON_AVAILABLE = True
except Exception:
    BIOPYTHON_AVAILABLE = False


# --- Standard amino acid order for one-hot encoding ---
AA_ORDER = [
    "ALA", "ARG", "ASN", "ASP", "CYS",
    "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO",
    "SER", "THR", "TRP", "TYR", "VAL",
]
AA_TO_INDEX = {aa: i for i, aa in enumerate(AA_ORDER)}
# One-hot encoding means exactly one position is 1.0 and all others are 0.0.
# Example: residue "HIS" -> index 8 -> vector[8] = 1.0, all other entries = 0.0.

# --- Simple residue chemistry classes ---
NEGATIVE = {"ASP", "GLU"}
POSITIVE = {"ARG", "LYS", "HIS"} #TODO: should I think somehow differentiate HIS from here since not so pos as others
POLAR = {"ASN", "ASP", "GLN", "GLU", "HIS", "LYS", "ARG", "SER", "THR", "TYR", "CYS"}
AROMATIC = {"PHE", "TYR", "TRP", "HIS"}
SULFUR = {"CYS", "MET"}
DONOR_CAPABLE = {"ARG", "ASN", "GLN", "HIS", "LYS", "SER", "THR", "TYR", "TRP", "CYS"}
ACCEPTOR_CAPABLE = {"ASP", "GLU", "ASN", "GLN", "HIS", "SER", "THR", "TYR", "CYS"}

# --- Backbone atom names used for simple BB/SC splitting later if needed ---
BACKBONE_ATOMS = {"N", "CA", "C", "O", "OXT"}

# --- Side-chain donor atoms relevant to metal binding / second-shell logic ---
# Keep K <= 2 for stage-1 simplicity.
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

# --- Stage-1 defaults --- #TODO: I think is too generalized and need other method
# Distance cutoffs are stage-1 defaults and should be tuned on validation data.
# Typical use: first-shell around direct coordination range, second-shell larger.
DEFAULT_FIRST_SHELL_CUTOFF = 3.0
DEFAULT_SECOND_SHELL_CUTOFF = 4.5
DEFAULT_POCKET_RADIUS = 8.0
DEFAULT_EDGE_RADIUS = 6.0
# Keep these constants centralized so they are easy to sweep/tune later.

# ============================================================
# Part 2: Basic tensor / geometry helper functions
# These helpers keep vector math safe and reusable.
# ============================================================
# Example input (Part 2):
#   1) x = tensor([[0., 0., 0.], [1., 0., 0.]])  # two 3D points
#   2) dim = -1 for xyz-norms
# Example output (Part 2):
#   1) safe_norm(x, dim=-1) -> tensor([eps, 1.0]) up to eps handling
#   2) pairwise_distances(x) -> tensor([[0., 1.], [1., 0.]])
# Here, x is a tensor of vectors or coordinates.
def safe_norm(x: Tensor, dim: int = -1, keepdim: bool = False, eps: float = 1e-8) -> Tensor:
    """Return a numerically stable Euclidean norm."""
    return torch.sqrt(torch.clamp((x * x).sum(dim=dim, keepdim=keepdim), min=eps))

def normalize_vec(x: Tensor, dim: int = -1, eps: float = 1e-8) -> Tensor:
    """Return x normalized to unit length, with numerical safety."""
    return x / safe_norm(x, dim=dim, keepdim=True, eps=eps)


def pairwise_distances(x: Tensor) -> Tensor:
    """Compute a full pairwise distance matrix for x with shape [N, 3]."""
    diff = x[:, None, :] - x[None, :, :]
    return safe_norm(diff, dim=-1)


def one_hot_index(index: int, size: int) -> Tensor:
    """Return a float one-hot vector of given size."""
    one_hot = torch.zeros(size, dtype=torch.float32)
    if 0 <= index < size:
        one_hot[index] = 1.0
    return one_hot


# ============================================================
# Part 3: Simple dataclasses describing residues and pockets
# These are plain Python containers that make preprocessing
# explicit and easier to debug before conversion into PyG Data.
# ============================================================
# Example input (Part 3):
#   1) chain_id="A", resseq=45, resname="HIS", atoms={"CA": tensor([..]), ...}
#   2) metal_coord=tensor([0., 0., 0.]) with a list of residues
# Example output (Part 3):
#   1) ResidueRecord for one residue
#   2) PocketRecord holding all residues around one Zn site

@dataclass
class ResidueRecord:
    """
    One residue with atom coordinates already collected.

    atoms:
        dict atom_name -> torch.tensor([x, y, z])

    plddt:
        For AF-derived structures, you can store residue mean pLDDT here.
        For experimental structures or missing values, set to 100 or 0 as needed.
    """
    chain_id: str
    resseq: int
    icode: str
    resname: str
    atoms: Dict[str, Tensor]
    plddt: float = 100.0

    # Fields filled later
    esm_embedding: Optional[Tensor] = None
    is_first_shell: bool = False
    is_second_shell: bool = False
    wall_distance: float = 0.0
    is_wall_residue: bool = False

    def residue_id(self) -> Tuple[str, int, str]:
        return (self.chain_id, self.resseq, self.icode)
    # What is icode?

    def get_atom(self, name: str) -> Optional[Tensor]:
        return self.atoms.get(name)

    def ca(self) -> Optional[Tensor]:
        return self.get_atom("CA")


@dataclass
class PocketRecord:
    """
    A single metal-centered pocket.

    residues:
        List of residues included in the pocket graph.

    metal_coord:
        Coordinate of the central metal ion for this pocket.

    y_metal / y_ec:
        Integer labels for training. You can leave them as None for inference-only datasets.
    """
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
# This section converts raw atom dictionaries into compact
# residue-level features suitable for a residue graph.
# ============================================================
# Example input (Part 4):
#   1) rr.resname = "HIS"
#   2) atoms include CA plus donor atoms (ND1/NE2)
# Example output (Part 4):
#   1) x_reschem vector with 6 metal-relevant chemistry features
#   2) donor_coords [K,3], donor_mask [K], and FG centroid [3]

def residue_one_hot(resname: str) -> Tensor:
    """Return a 20D one-hot over standard amino acids. Unknown residues become all-zero."""
    idx = AA_TO_INDEX.get(resname, -1)
    return one_hot_index(idx, len(AA_ORDER))


def residue_charge_class(resname: str) -> Tensor:
    """
    Return a 3D one-hot for charge class:
    [negative, neutral, positive]
    """
    if resname in NEGATIVE:
        return torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
    if resname in POSITIVE:
        return torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
    return torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)


def residue_chemistry_flags(resname: str) -> Tensor:
    """
    Return the lean chemistry flags used with ESM embeddings:
    donor-capable, acceptor-capable, sulfur, acidic, basic, histidine
    """
    flags = [
        float(resname in DONOR_CAPABLE),
        float(resname in ACCEPTOR_CAPABLE),
        float(resname in SULFUR),
        float(resname in NEGATIVE),
        float(resname in POSITIVE),
        float(resname == "HIS"),
    ]
    return torch.tensor(flags, dtype=torch.float32)


def build_x_reschem(residue: ResidueRecord) -> Tensor:
    """
    Build a compact handcrafted chemistry feature vector.

    Output dimension:
        6 (chem flags)
    """
    return residue_chemistry_flags(residue.resname)


def donor_atom_names(resname: str) -> List[str]:
    """Return up to two donor-like atoms for the residue."""
    atoms = DONOR_ATOMS_BY_RESIDUE.get(resname, [])
    return atoms[:2]


def donor_coords_and_mask(residue: ResidueRecord, max_donors: int = 2) -> Tuple[Tensor, Tensor]:
    """
    Return donor coordinates [K, 3] and mask [K] for a residue.

    Missing donor slots are filled with zeros and mask=False.
    """
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
    """Return side-chain atom coordinates only."""
    sidechain = []
    for atom_name, coord in residue.atoms.items():
        if atom_name not in BACKBONE_ATOMS:
            sidechain.append(coord.float())
    return sidechain


def centroid(coords: List[Tensor]) -> Optional[Tensor]:
    """Return the centroid of a list of coordinates, or None if empty."""
    if len(coords) == 0:
        return None
    return torch.stack(coords, dim=0).mean(dim=0)


def functional_group_centroid(residue: ResidueRecord) -> Tensor:
    """
    Return a chemically meaningful residue-local centroid.

    Priority:
    1) donor atom centroid if donor atoms exist
    2) side-chain centroid if side-chain atoms exist
    3) CA coordinate
    """
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
    """Return the minimum distance from a set of coords [K,3] to a single point [3]."""
    if coords.numel() == 0:
        return 999.0
    if mask is not None:
        coords = coords[mask]
    if coords.numel() == 0:
        return 999.0
    return float(safe_norm(coords - point.unsqueeze(0), dim=-1).min().item())


def second_min_distance_to_point(coords: Tensor, point: Tensor, mask: Optional[Tensor] = None) -> float:
    """Return the second-smallest distance from coords [K,3] to point [3], or the smallest if only one exists."""
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
# This is useful if you want a direct route from structure file
# to PocketRecord, but it stays intentionally minimal.
# ============================================================
# Example input (Part 5):
#   1) filepath="1abc.cif"
#   2) pocket_radius=8.0 for residue inclusion
# Example output (Part 5):
#   1) Biopython structure object (from parse_structure_file)
#   2) list[PocketRecord], one entry per detected Zn center

def parse_structure_file(filepath: str, structure_id: Optional[str] = None):
    """
    Parse a PDB or mmCIF file with Biopython and return the structure object.

    This function is optional. If you already have your own parser, you can skip it.
    """
    if not BIOPYTHON_AVAILABLE:
        raise ImportError("Biopython is not installed. Run: pip install biopython")

    path = Path(filepath)
    sid = structure_id or path.stem

    if path.suffix.lower() in {".cif", ".mmcif"}:
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)

    return parser.get_structure(sid, str(path))


def residue_record_from_biopython_residue(residue) -> Optional[ResidueRecord]:
    """
    Convert a Biopython residue into ResidueRecord.

    Returns None for residues without CA.
    """
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
    """
    Extract one PocketRecord per Zn ion found in the structure.

    Pocket inclusion rule:
        residue has any atom within pocket_radius Å of the Zn atom
    """
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
# Part 6: External ESM embedding attachment
# ESM3 embeddings are not computed in this script; instead, this
# helper attaches already-computed per-residue embeddings.
# ============================================================
# Example input (Part 6):
#   1) esm_lookup[(chain_id, resseq, icode)] = tensor([D_esm])
#   2) pocket with residues carrying matching IDs
# Example output (Part 6):
#   1) rr.esm_embedding set from lookup for matched residues
#   2) zero vector fallback for missing residues (if zero_if_missing=True)

def attach_esm_embeddings(
    pocket: PocketRecord,
    esm_lookup: Dict[Tuple[str, int, str], Tensor],
    esm_dim: int,
    zero_if_missing: bool = True,
) -> None:
    """
    Attach a per-residue ESM embedding to each residue in the pocket.

    esm_lookup key:
        (chain_id, resseq, icode)

    If a residue is missing and zero_if_missing=True, a zero vector is used.
    """
    for rr in pocket.residues:
        key = rr.residue_id()
        if key in esm_lookup:
            rr.esm_embedding = esm_lookup[key].float()
        elif zero_if_missing:
            rr.esm_embedding = torch.zeros(esm_dim, dtype=torch.float32)
        else:
            raise KeyError(f"Missing ESM embedding for residue key {key}")


# ============================================================
# Part 7: Shell annotation and simple wall placeholders
# These functions compute first-shell / second-shell flags from
# donor coordinates and can be expanded later.
# ============================================================
# Example input (Part 7):
#   1) pocket with metal_coord + residue donor atoms
#   2) first_shell_cutoff=3.0, second_shell_cutoff=4.0
# Example output (Part 7):
#   1) rr.is_first_shell and rr.is_second_shell assigned per residue
#   2) rr.wall_distance and rr.is_wall_residue assigned (naive placeholder)

def annotate_shell_roles(
    pocket: PocketRecord,
    first_shell_cutoff: float = DEFAULT_FIRST_SHELL_CUTOFF,
    second_shell_cutoff: float = DEFAULT_SECOND_SHELL_CUTOFF,
) -> None:
    """
    Mark residues as first-shell or second-shell.

    First shell:
        any donor atom within first_shell_cutoff Å of the metal

    Second shell:
        not first-shell, but within second_shell_cutoff Å of any first-shell
        functional centroid
    """
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


def annotate_wall_features_naive(
    pocket: PocketRecord,
    pocket_radius: float = DEFAULT_POCKET_RADIUS,
) -> None:
    """
    Very simple placeholder wall feature.

    Idea:
        residues closer to the outer pocket radius are more 'wall-like'.

    This is intentionally naive for stage-1. You can later replace it
    with a proper pocket-surface or cavity method.
    """
    metal = pocket.metal_coord.float()

    for rr in pocket.residues:
        ca = rr.ca()
        if ca is None:
            rr.wall_distance = pocket_radius
            rr.is_wall_residue = False
            continue

        d = float(safe_norm(ca - metal, dim=-1).item())
        rr.wall_distance = max(0.0, pocket_radius - d)
        rr.is_wall_residue = bool(d >= 0.75 * pocket_radius)


# ============================================================
# Part 8: Stage-1 graph construction
# This is the key preprocessing section: it converts a pocket
# into one torch_geometric.data.Data object.
# ============================================================
# Example input (Part 8):
#   1) one PocketRecord with N residues and metal coordinate
#   2) esm_dim=256 and edge_radius=10.0
# Example output (Part 8):
#   1) PyG Data node tensors (x_esm, x_reschem, x_role, x_dist_raw, x_misc, x_vec)
#   2) PyG edge tensors (edge_index, edge_dist_raw) plus labels if available

def build_radius_graph(pos: Tensor, radius: float) -> Tensor:
    """
    Build a directed radius graph using pure PyTorch.

    Returns:
        edge_index [2, E]
    """
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
    """
    Compute the minimum donor/functional-group distance between two residues.

    For stage-1 simplicity:
      - use donor atoms if available
      - otherwise use functional-group centroids
    """
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


def residue_to_stage1_node_features(rr: ResidueRecord, metal_coord: Tensor, esm_dim: int) -> Dict[str, Tensor]:
    """
    Convert one ResidueRecord into the stage-1 node feature tensors.

    Outputs:
        x_esm      [D_esm]
        x_reschem  [6]
        x_role     [2]
        x_dist_raw [4]
        x_misc     [3]
        x_vec      [1, 3]
        donor_coords [2, 3]
        donor_mask [2]
        fg_centroid [3]
        pos [3]
    """
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
        dtype=torch.float32
    )

    x_dist_raw = torch.tensor(
        [ca_to_metal, fg_to_metal, min_donor_to_metal, second_min_donor_to_metal],
        dtype=torch.float32
    )

    x_misc = torch.tensor(
        [float(rr.plddt), float(rr.wall_distance), float(rr.is_wall_residue)],
        dtype=torch.float32
    )

    # Residue-local vector feature:
    # functional-group centroid minus CA, shape [1, 3]
    x_vec = (fg - ca).unsqueeze(0).float()

    return {
        "x_esm": rr.esm_embedding.float(),
        "x_reschem": build_x_reschem(rr).float(),
        "x_role": x_role,
        "x_dist_raw": x_dist_raw,
        "x_misc": x_misc,
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
    """
    Convert one PocketRecord into a PyG Data object for the stage-1 model.

    Stage-1 design:
      - rich node features
      - simple geometric edges
      - no explicit chemistry-aware edge flags yet
    """
    annotate_shell_roles(pocket)
    annotate_wall_features_naive(pocket)

    node_dicts = [residue_to_stage1_node_features(rr, pocket.metal_coord, esm_dim) for rr in pocket.residues]
    N = len(node_dicts)

    x_esm = torch.stack([d["x_esm"] for d in node_dicts], dim=0)
    x_reschem = torch.stack([d["x_reschem"] for d in node_dicts], dim=0)
    x_role = torch.stack([d["x_role"] for d in node_dicts], dim=0)
    x_dist_raw = torch.stack([d["x_dist_raw"] for d in node_dicts], dim=0)
    x_misc = torch.stack([d["x_misc"] for d in node_dicts], dim=0)
    x_vec = torch.stack([d["x_vec"] for d in node_dicts], dim=0)
    donor_coords = torch.stack([d["donor_coords"] for d in node_dicts], dim=0)
    donor_mask = torch.stack([d["donor_mask"] for d in node_dicts], dim=0)
    fg_centroid = torch.stack([d["fg_centroid"] for d in node_dicts], dim=0)
    pos = torch.stack([d["pos"] for d in node_dicts], dim=0)

    edge_index = build_radius_graph(pos, edge_radius)

    # If the graph has no edges, connect nearest neighbors minimally to avoid dead graphs.
    if edge_index.size(1) == 0 and N > 1: #TODO:If that the case  preffer the code will errored imidiately and will not use this solution
        dmat = pairwise_distances(pos)
        dmat = dmat + torch.eye(N) * 1e6
        src = torch.arange(N, dtype=torch.long)
        dst = dmat.argmin(dim=1)
        edge_index = torch.stack([src, dst], dim=0)

    src, dst = edge_index
    ca_ca_dist = safe_norm(pos[dst] - pos[src], dim=-1).unsqueeze(-1)

    fg_fg = []
    for i, j in zip(src.tolist(), dst.tolist()):
        fg_fg.append(min_fg_fg_distance(pocket.residues[i], pocket.residues[j]))
    fg_fg_dist = torch.tensor(fg_fg, dtype=torch.float32).unsqueeze(-1)

    # Keep exactly 2 raw edge distances in stage-1
    edge_dist_raw = torch.cat([ca_ca_dist, fg_fg_dist], dim=-1)   # [E, 2] #TODO: Add to the pipeline overall closest atoms to each fg from the other residue.

    y_metal = None if pocket.y_metal is None else torch.tensor([pocket.y_metal], dtype=torch.long) #TODO: what is going here. what should I add?
    y_ec = None if pocket.y_ec is None else torch.tensor([pocket.y_ec], dtype=torch.long)

    data = Data(
        x_esm=x_esm,
        x_reschem=x_reschem,
        x_role=x_role,
        x_dist_raw=x_dist_raw,
        x_misc=x_misc,
        x_vec=x_vec,
        pos=pos,
        fg_centroid=fg_centroid,
        donor_coords=donor_coords,
        donor_mask=donor_mask,
        edge_index=edge_index,
        edge_dist_raw=edge_dist_raw,
        zinc_pos=pocket.metal_coord.unsqueeze(0),  # stored as [1,3] for consistency
    )

    # TODO: From where is the y_ come from? what is going here. what should I add?
    if y_metal is not None:
        data.y_metal = y_metal
    if y_ec is not None:
        data.y_ec = y_ec

    return data


# ============================================================
# Part 9: Dataset wrapper
# This thin wrapper stores PocketRecord objects and converts each
# one into PyG Data on demand.
# ============================================================
# Example input (Part 9):
#   1) pockets=[p0, p1, p2, ...]
#   2) esm_dim=256
# Example output (Part 9):
#   1) len(dataset) equals number of pockets
#   2) dataset[i] returns one fully built PyG Data graph

class PocketGraphDataset(Dataset):
    """
    Dataset of PocketRecord objects converted to PyG Data.
    """
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
# Part 10: RBF expansion module
# This turns raw distances into smoother basis features, which
# is usually easier for graph models to learn from.
# ============================================================
# Example input (Part 10):
#   1) d = edge_dist_raw with shape [E, 2]
#   2) n_rbf=16 radial centers in [d_min, d_max]
# Example output (Part 10):
#   1) expanded basis tensor shape [E, 2, 16]
#   2) downstream encoder flattens this to [E, 32]

class RBFExpansion(nn.Module):
    """
    Expand one or more scalar distances into radial basis functions.
    """
    def __init__(self, n_rbf: int = 16, d_min: float = 0.0, d_max: float = 12.0):
        super().__init__()
        centers = torch.linspace(d_min, d_max, n_rbf)
        self.register_buffer("centers", centers)
        width = (d_max - d_min) / n_rbf
        self.gamma = 1.0 / (width * width + 1e-8)

    def forward(self, d: Tensor) -> Tensor:
        # d: [..., n_dist]
        return torch.exp(-self.gamma * (d.unsqueeze(-1) - self.centers) ** 2)


# ============================================================
# Part 11: Stage-1 encoders for node and edge scalars
# These modules turn raw handcrafted inputs + ESM embeddings into
# hidden scalar channels for the graph model.
# ============================================================
# Example input (Part 11):
#   1) node scalars: x_esm [N,256], x_reschem [N,6], x_role [N,2], x_dist_raw [N,4], x_misc [N,3]
#   2) edge scalars: edge_dist_raw [E,2]
# Example output (Part 11):
#   1) NodeScalarEncoder -> s0 [N, hidden_s]
#   2) EdgeScalarEncoder -> edge_s [E, edge_hidden]

class NodeScalarEncoder(nn.Module):
    """
    Encode raw node scalar inputs into hidden scalar channels.

    Inputs:
      - x_esm      [N, D_esm]
      - x_reschem  [N, 6]
      - x_role     [N, 2]
      - x_dist_raw [N, 4]
      - x_misc     [N, 3]

    Distances are RBF-expanded internally.
    """
    def __init__(self, esm_dim: int, esm_proj: int = 128, n_rbf: int = 16, out_dim: int = 128):
        super().__init__()
        self.esm_proj = nn.Sequential(
            nn.Linear(esm_dim, esm_proj),
            nn.LayerNorm(esm_proj),
            nn.SiLU(),
            nn.Dropout(0.1),
        )
        self.dist_rbf = RBFExpansion(n_rbf=n_rbf, d_min=0.0, d_max=12.0)

        # 6 (chem) + 2 (role) + 3 (misc) + 4*n_rbf (RBF distance) + esm_proj
        in_dim = esm_proj + 6 + 2 + 3 + 4 * n_rbf

        self.out_proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.SiLU(),
        )

    def forward(self, x_esm: Tensor, x_reschem: Tensor, x_role: Tensor, x_dist_raw: Tensor, x_misc: Tensor) -> Tensor:
        esm = self.esm_proj(x_esm)
        d_rbf = self.dist_rbf(x_dist_raw).flatten(start_dim=1)
        x = torch.cat([esm, x_reschem, x_role, x_misc, d_rbf], dim=-1)
        return self.out_proj(x)


class EdgeScalarEncoder(nn.Module):
    """
    Encode stage-1 raw edge distances into hidden edge scalar channels.

    Inputs:
      - edge_dist_raw [E, 2]
          0 = CA-CA distance
          1 = FG-FG minimum distance
    """
    def __init__(self, n_rbf: int = 16, out_dim: int = 64):
        super().__init__()
        self.dist_rbf = RBFExpansion(n_rbf=n_rbf, d_min=0.0, d_max=12.0)
        in_dim = 2 * n_rbf
        self.out_proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.SiLU(),
        )

    def forward(self, edge_dist_raw: Tensor) -> Tensor:
        d_rbf = self.dist_rbf(edge_dist_raw).flatten(start_dim=1)
        return self.out_proj(d_rbf)


# ============================================================
# Part 12: Simplified GVP-style blocks
# These are the core geometric layers. They keep separate scalar
# and vector channels and mix them in a controlled way.
# ============================================================
# Example input (Part 12):
#   1) node scalar state s [N, S] and node vector state v [N, V, 3]
#   2) edge_index [2, E], edge_s [E, E_dim], edge_v [E, 1, 3]
# Example output (Part 12):
#   1) message/update pass returns s_out [N, S]
#   2) message/update pass returns v_out [N, V, 3]

def vector_norm(v: Tensor, eps: float = 1e-8) -> Tensor:
    """
    Compute per-vector-channel norms.

    Input:
        v [N, V, 3]

    Output:
        [N, V]
    """
    return torch.sqrt(torch.clamp((v * v).sum(dim=-1), min=eps))


class SimpleGVP(nn.Module):
    """
    A compact GVP-style unit.

    What it does:
    - converts vector norms into scalar information
    - updates scalar channels using scalar + vector-norm inputs
    - linearly mixes vector channels
    - gates vector outputs using the new scalar channels

    This is intentionally simpler than full published GVP code,
    but it preserves the scalar/vector separation you want.
    """
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
        # Scalar update uses scalar inputs plus vector norms
        v_norm = vector_norm(v)
        s_cat = torch.cat([s, v_norm], dim=-1)
        s_out = self.scalar_mlp(s_cat)

        # Vector update mixes channels but keeps xyz geometry
        v_t = v.transpose(1, 2)        # [N, 3, V_in]
        v_proj = self.vector_linear(v_t).transpose(1, 2)   # [N, V_out, 3]

        # Scalar-controlled gating on vector channels
        gate = torch.sigmoid(self.vector_gate(s_out)).unsqueeze(-1)
        v_out = v_proj * gate

        return s_out, v_out


class SimpleGVPLayer(nn.Module):
    """
    One message-passing layer for the residue graph.

    Inputs:
      - node scalar state s
      - node vector state v
      - edge scalar state edge_s
      - edge vector state edge_v
      - edge_index

    Operation:
      1) create source-to-destination messages
      2) aggregate messages at each destination node
      3) apply a residual update to node states
    """
    def __init__(self, s_dim: int, v_dim: int, e_dim: int):
        super().__init__()

        self.message_gvp = SimpleGVP(
            s_in=2 * s_dim + e_dim + 1,   # src scalar + dst scalar + edge scalar + edge length
            v_in=2 * v_dim + 1,           # src vec + dst vec + edge vec
            s_out=s_dim,
            v_out=v_dim,
        )

        self.update_gvp = SimpleGVP(
            s_in=2 * s_dim,               # current scalar + aggregated scalar
            v_in=2 * v_dim,               # current vector + aggregated vector
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

        edge_len = vector_norm(edge_v)  # [E, 1]

        # Message scalar input
        m_s_in = torch.cat([s_src, s_dst, edge_s, edge_len], dim=-1)

        # Message vector input
        m_v_in = torch.cat([v_src, v_dst, edge_v], dim=1)

        m_s, m_v = self.message_gvp(m_s_in, m_v_in)

        # Aggregate scalar messages using index_add_
        agg_s = torch.zeros_like(s)
        agg_s.index_add_(0, dst, m_s)

        # Aggregate vector messages using index_add_ over nodes
        agg_v = torch.zeros_like(v)
        agg_v.index_add_(0, dst, m_v)

        # Update node states with residual-style block
        u_s_in = torch.cat([s, agg_s], dim=-1)
        u_v_in = torch.cat([v, agg_v], dim=1)
        ds, dv = self.update_gvp(u_s_in, u_v_in)

        s_out = self.norm_s(s + ds)
        v_out = v + dv
        return s_out, v_out


# ============================================================
# Part 13: Full stage-1 graph classifier
# This model combines the encoders, the GVP-style stack, and the
# final graph-level heads for metal + EC classification.
# ============================================================
# Example input (Part 13):
#   1) batched Data object from PyG DataLoader
#   2) data.batch marks node-to-graph assignments
# Example output (Part 13):
#   1) outputs["logits_metal"] with shape [B, n_metal]
#   2) outputs["logits_ec"] with shape [B, n_ec]
#   3) outputs["embed"] and optional outputs["loss"] when labels exist

class GVPPocketClassifier(nn.Module):
    """
    Stage-1 pocket classifier.

    Design:
      - rich scalar node features
      - one residue-local vector feature
      - simple geometric edges
      - multiple GVP-style layers
      - graph-level readout from pooled scalar states
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
    ):
        super().__init__()

        self.node_scalar_encoder = NodeScalarEncoder(
            esm_dim=esm_dim,
            esm_proj=128,
            n_rbf=16,
            out_dim=hidden_s,
        )

        self.edge_scalar_encoder = EdgeScalarEncoder(
            n_rbf=16,
            out_dim=edge_hidden,
        )

        # Project the initial single vector channel [N,1,3] to hidden_v channels
        self.init_vec_proj = nn.Linear(1, hidden_v, bias=False)

        self.layers = nn.ModuleList([
            SimpleGVPLayer(
                s_dim=hidden_s,
                v_dim=hidden_v,
                e_dim=edge_hidden,
            )
            for _ in range(n_layers)
        ])

        self.head_metal = nn.Sequential(
            nn.Linear(2 * hidden_s, hidden_s),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_s, n_metal),
        )

        self.head_ec = nn.Sequential(
            nn.Linear(2 * hidden_s, hidden_s),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_s, n_ec),
        )

        self.ce = nn.CrossEntropyLoss()

    def _init_vector_channels(self, x_vec: Tensor) -> Tensor:
        """
        Convert [N,1,3] initial node vectors into hidden vector channels [N,V,3].
        """
        x_t = x_vec.transpose(1, 2)         # [N, 3, 1]
        x_proj = self.init_vec_proj(x_t)    # [N, 3, hidden_v]
        return x_proj.transpose(1, 2)       # [N, hidden_v, 3]

    def _prepare_edge_vectors(self, edge_index: Tensor, pos: Tensor) -> Tensor:
        """
        Build normalized edge relative vectors from node positions.

        Output shape:
            [E, 1, 3]
        """
        src, dst = edge_index
        rel = pos[dst] - pos[src]
        rel = normalize_vec(rel, dim=-1)
        return rel.unsqueeze(1)

    def forward(self, data: Data) -> Dict[str, Tensor]:
        # Encode scalar node features
        s = self.node_scalar_encoder(
            data.x_esm,
            data.x_reschem,
            data.x_role,
            data.x_dist_raw,
            data.x_misc,
        )

        # Initialize vector node channels
        v = self._init_vector_channels(data.x_vec)

        # Encode scalar edge features
        edge_s = self.edge_scalar_encoder(data.edge_dist_raw)

        # Prepare geometric edge vectors from positions
        edge_v = self._prepare_edge_vectors(data.edge_index, data.pos)

        # Message passing stack
        for layer in self.layers:
            s, v = layer(s, v, data.edge_index, edge_s, edge_v)

        # Graph-level readout uses pooled scalar channels only
        pooled_mean = global_mean_pool(s, data.batch)
        pooled_max = global_max_pool(s, data.batch)
        pocket_embed = torch.cat([pooled_mean, pooled_max], dim=-1)

        logits_metal = self.head_metal(pocket_embed)
        logits_ec = self.head_ec(pocket_embed)

        outputs = {
            "logits_metal": logits_metal,
            "logits_ec": logits_ec,
            "embed": pocket_embed,
        }

        if hasattr(data, "y_metal") and hasattr(data, "y_ec"):
            loss = self.ce(logits_metal, data.y_metal) + self.ce(logits_ec, data.y_ec)
            outputs["loss"] = loss

        return outputs


# ============================================================
# Part 14: Training and evaluation utilities
# These are intentionally compact and easy to expand later.
# ============================================================
# Example input (Part 14):
#   1) model, DataLoader, optimizer, device
#   2) batches include labels y_metal and y_ec for training
# Example output (Part 14):
#   1) train_epoch returns mean scalar loss for one epoch
#   2) predict_batch returns concatenated logits (and labels when present)

def train_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, device: str = "cpu") -> float:
    """Train one epoch and return mean loss."""
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
    """
    Run inference over a loader and return concatenated logits/labels.
    """
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
    """Compute simple accuracy from logits and integer labels."""
    pred = logits.argmax(dim=-1)
    return float((pred == y).float().mean().item())


# ============================================================
# Part 15: Synthetic smoke-test data generation
# This gives you a way to verify shapes and a forward pass even
# before you connect real structures and real ESM embeddings.
# ============================================================
# Example input (Part 15):
#   1) synthetic_pocket("p0", n_residues=14, esm_dim=256)
#   2) random seed optionally set for reproducibility
# Example output (Part 15):
#   1) PocketRecord with synthetic atoms and embeddings
#   2) random classification labels for smoke testing only

def random_residue(resname: Optional[str] = None, center: Optional[Tensor] = None) -> ResidueRecord:
    """
    Create a fake residue with a minimal plausible atom set for smoke testing.
    """
    if resname is None:
        resname = random.choice(AA_ORDER)
    if center is None:
        center = torch.randn(3) * 3.0

    atoms = {}
    # Minimal backbone
    atoms["CA"] = center + torch.randn(3) * 0.2
    atoms["N"] = center + torch.tensor([-1.2, 0.3, 0.0]) + torch.randn(3) * 0.1
    atoms["C"] = center + torch.tensor([1.2, -0.2, 0.0]) + torch.randn(3) * 0.1
    atoms["O"] = center + torch.tensor([1.8, -0.5, 0.1]) + torch.randn(3) * 0.1

    # Add donor-like atoms if the residue has them
    for atom_name in donor_atom_names(resname):
        atoms[atom_name] = center + torch.randn(3) * 0.6 + torch.tensor([0.5, 0.5, 0.5])

    # Add a generic side-chain centroid atom if needed
    if all(a in BACKBONE_ATOMS for a in atoms.keys()):
        atoms["CB"] = center + torch.tensor([0.3, 1.1, 0.2]) + torch.randn(3) * 0.1

    return ResidueRecord(
        chain_id="A",
        resseq=random.randint(1, 999),
        icode="",
        resname=resname,
        atoms=atoms,
        plddt=float(random.uniform(70.0, 100.0)),
    )


def synthetic_pocket(
    pocket_id: str,
    n_residues: int,
    esm_dim: int,
    n_metal_classes: int = 8,
    n_ec_classes: int = 7,
) -> PocketRecord:
    """
    Build a synthetic pocket for debugging only.
    """
    metal = torch.zeros(3, dtype=torch.float32)

    residues = []
    for _ in range(n_residues):
        center = torch.randn(3) * 3.0
        rr = random_residue(center=center)
        rr.esm_embedding = torch.randn(esm_dim)
        residues.append(rr)

    pocket = PocketRecord(
        structure_id="synthetic",
        pocket_id=pocket_id,
        metal_element="ZN",
        metal_coord=metal,
        residues=residues,
        y_metal=random.randint(0, n_metal_classes - 1),
        y_ec=random.randint(0, n_ec_classes - 1),
    )
    return pocket


# ============================================================
# Part 16: A complete runnable smoke test
# This is the easiest way to verify that the whole pipeline
# works before you connect real data.
# ============================================================
# Example input (Part 16):
#   1) run_smoke_test(device="cpu")
#   2) no external files required
# Example output (Part 16):
#   1) printed train loss for one epoch
#   2) printed logits shapes and simple synthetic accuracies

def run_smoke_test(device: str = "cpu") -> None:
    """
    Create a few synthetic graphs, batch them, and run one train step.
    """
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
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

    # One train epoch
    loss = train_epoch(model, loader, optimizer, device=device)
    print(f"Smoke-test train loss: {loss:.4f}")

    # One evaluation pass
    result = predict_batch(model, loader, device=device)
    print("Metal logits shape:", tuple(result["metal_logits"].shape))
    print("EC logits shape:", tuple(result["ec_logits"].shape))
    print("Metal acc (random synthetic):", accuracy_from_logits(result["metal_logits"], result["metal_y"]))
    print("EC acc (random synthetic):", accuracy_from_logits(result["ec_logits"], result["ec_y"]))
    print("Smoke test completed successfully.")


# ============================================================
# Part 17: Optional helpers for saving and loading preprocessed
# pocket metadata. These are convenience functions only.
# ============================================================
# Example input (Part 17):
#   1) save_pocket_metadata_json(pocket, "pocket.json")
#   2) pocket already annotated/constructed in memory
# Example output (Part 17):
#   1) JSON file with pocket-level fields
#   2) residue list with IDs, labels, and atom-name inventory

def save_pocket_metadata_json(pocket: PocketRecord, outpath: str) -> None:
    """
    Save lightweight pocket metadata to JSON.

    This does not save tensors like ESM embeddings or coordinates in a full
    binary-efficient way; it is mainly for debugging pocket composition.
    """
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
                "wall_distance": rr.wall_distance,
                "is_wall_residue": rr.is_wall_residue,
                "atom_names": sorted(list(rr.atoms.keys())),
            }
            for rr in pocket.residues
        ],
    }

    with open(outpath, "w") as f:
        json.dump(payload, f, indent=2)


# ============================================================
# Part 18: Main entrypoint
# If you run this file directly, it performs the smoke test.
# ============================================================
# Example input (Part 18):
#   1) python main_v2.py
# Example output (Part 18):
#   1) the Part-16 smoke test runs end-to-end
#   2) logs are printed to stdout

if __name__ == "__main__":
    run_smoke_test(device="cpu")
