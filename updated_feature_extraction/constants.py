from __future__ import annotations

FEATURE_NAMES = (
    "SASA",
    "BSA",
    "SolvEnergy",
    "fa_sol",
    "pKa_shift",
    "dpKa_desolv",
    "dpKa_bg",
    "dpKa_titr",
    "omega",
    "rama_prepro",
    "fa_dun",
    "fa_elec",
    "fa_atr",
    "fa_rep",
)

# Maximum solvent-accessible surface areas for amino acids.
# Values are from a common Tien et al.-style reference table and are used
# only to build bounded burial proxies from observed SASA.
MAX_RESIDUE_SASA = {
    "ALA": 121.0,
    "ARG": 265.0,
    "ASN": 187.0,
    "ASP": 187.0,
    "CYS": 148.0,
    "GLN": 214.0,
    "GLU": 214.0,
    "GLY": 97.0,
    "HIS": 216.0,
    "ILE": 195.0,
    "LEU": 191.0,
    "LYS": 230.0,
    "MET": 203.0,
    "PHE": 228.0,
    "PRO": 154.0,
    "SER": 143.0,
    "THR": 163.0,
    "TRP": 264.0,
    "TYR": 255.0,
    "VAL": 165.0,
}

KYTE_DOOLITTLE_HYDROPATHY = {
    "ALA": 1.8,
    "ARG": -4.5,
    "ASN": -3.5,
    "ASP": -3.5,
    "CYS": 2.5,
    "GLN": -3.5,
    "GLU": -3.5,
    "GLY": -0.4,
    "HIS": -3.2,
    "ILE": 4.5,
    "LEU": 3.8,
    "LYS": -3.9,
    "MET": 1.9,
    "PHE": 2.8,
    "PRO": -1.6,
    "SER": -0.8,
    "THR": -0.7,
    "TRP": -0.9,
    "TYR": -1.3,
    "VAL": 4.2,
}

FORMAL_RESIDUE_CHARGES = {
    "ARG": 1.0,
    "ASP": -1.0,
    "GLU": -1.0,
    "HIS": 0.5,
    "LYS": 1.0,
}

METAL_FORMAL_CHARGES = {
    "CA": 2.0,
    "CO": 2.0,
    "CU": 2.0,
    "FE": 2.0,
    "K": 1.0,
    "MG": 2.0,
    "MN": 2.0,
    "NA": 1.0,
    "NI": 2.0,
    "ZN": 2.0,
}

BACKBONE_ATOM_NAMES = frozenset({"N", "CA", "C", "O", "OXT"})

RAMA_BASINS_DEGREES = {
    "general": [(-63.0, -42.0), (-120.0, 130.0), (60.0, 40.0)],
    "gly": [(-80.0, 0.0), (-120.0, 130.0), (75.0, 15.0)],
    "pro": [(-65.0, 145.0), (-75.0, -30.0)],
    "prepro": [(-75.0, 145.0), (-80.0, -25.0)],
}

ROTAMER_CENTERS_DEGREES = (-60.0, 60.0, 180.0)
