from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from torch import Tensor

AA_ORDER = [
    "ALA", "ARG", "ASN", "ASP", "CYS",
    "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO",
    "SER", "THR", "TRP", "TYR", "VAL",
]
AA_TO_INDEX = {aa: i for i, aa in enumerate(AA_ORDER)}

NEGATIVE = {"ASP", "GLU"}
POSITIVE = {"ARG", "LYS", "HIS"}
AROMATIC = {"PHE", "TYR", "TRP", "HIS"}
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
DEFAULT_POCKET_RADIUS = 8.0
DEFAULT_EDGE_RADIUS = 6.0
DEFAULT_MULTINUCLEAR_MERGE_DISTANCE = 4.5
GENERIC_METAL_ELEMENT = "METAL"

# Used only to detect generic transition-metal-centered sites in structures; the
# true metal identity should stay out of the model inputs when metal type is a
# prediction target.
SUPPORTED_SITE_METAL_ELEMENTS = {
    "SC", "TI", "V", "CR", "MN", "FE", "CO", "NI", "CU", "ZN",
    "Y", "ZR", "NB", "MO", "TC", "RU", "RH", "PD", "AG", "CD",
    "HF", "TA", "W", "RE", "OS", "IR", "PT", "AU", "HG",
}

NODE_FEATURES_CURRENTLY_HANDLED = [
    "aa_one_hot",
    "charge_class_3way",
    "donor_flag",
    "acceptor_flag",
    "aromatic_flag",
    "acidic_flag",
    "basic_flag",
    "is_first_shell",
    "is_second_shell",
    "ca_to_metal",
    "fg_to_metal",
    "min_donor_to_metal",
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
    "v_ca_to_fg",
    "v_res_to_metal",
    "cos_theta_between_vnetligand_to_vrestometal",
]

EDGE_FEATURES_RECOMMENDED_RING = [
    "ring_contact_type_one_hot",
    "contact_distance",
    "sequence_separation",
]

INTERACTION_SUMMARIES_OPTIONAL_WITH_RING = [
    "HBOND:MC_MC",
    "HBOND:SC_SC",
    "HBOND:MC_SC",
    "HBOND:SC_MC",
    "VDW:SC_SC",
    "VDW:MC_MC",
    "VDW:MC_SC",
    "VDW:SC_MC",
    "IONIC:SC_SC",
    "PIPISTACK:SC_SC",
    "PICATION:SC_SC",
    "METAL_ION:SC_LIG",
]
RING_INTERACTION_TYPES = set(INTERACTION_SUMMARIES_OPTIONAL_WITH_RING)
RING_INTERACTION_TO_INDEX = {
    interaction: i for i, interaction in enumerate(INTERACTION_SUMMARIES_OPTIONAL_WITH_RING)
}

EDGE_SOURCE_TYPES = ["radius", "ring"]
EDGE_SOURCE_TO_INDEX = {source: i for i, source in enumerate(EDGE_SOURCE_TYPES)}

NORMALIZABLE_FEATURE_NAMES = (
    "x_dist_raw",
    "x_misc",
    "x_env_burial",
    "x_env_pka",
    "x_env_conf",
    "x_env_interactions",
    "edge_dist_raw",
    "edge_seqsep",
    "site_metal_stats",
)


@dataclass
class ResidueRecord:
    chain_id: str
    resseq: int
    icode: str
    resname: str
    atoms: Dict[str, Tensor]
    esm_embedding: Optional[Tensor] = None
    is_first_shell: bool = False
    is_second_shell: bool = False
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
    metal_coords: List[Tensor] = field(default_factory=list)
    y_metal: Optional[int] = None
    y_ec: Optional[int] = None
    y_multinuclear: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def resolved_metal_coords(self) -> List[Tensor]:
        return self.metal_coords if self.metal_coords else [self.metal_coord]

    def metal_count(self) -> int:
        return len(self.resolved_metal_coords())

    def is_multinuclear(self) -> bool:
        return self.metal_count() > 1
