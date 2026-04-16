from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Optional, Tuple

from training.esm_feature_loading import ResidueKey


RELEVANT_FEATURE_NAMES = (
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

ENERGY_TERM_NAMES = ("fa_sol", "fa_elec", "omega", "rama_prepro", "fa_dun", "fa_atr", "fa_rep")

CHAIN_FROM_NAME_RE = re.compile(r"__chain_([^_]+)__")
ENERGY_LABEL_RE = re.compile(r"^(?P<resname>[A-Z0-9]{3})(?::[A-Za-z0-9]+)?_(?P<pose_idx>\d+)$")
BSA_REPORT_RE = re.compile(r"REPORT:\s+([A-Z0-9]{3})(-?\d+)([A-Za-z]?)\t([-+0-9.eE]+)")


def infer_chain_id(structure_name: str) -> str:
    match = CHAIN_FROM_NAME_RE.search(structure_name)
    if match:
        return match.group(1)
    return ""


def default_feature_dict() -> Dict[str, float]:
    features: Dict[str, float] = {}
    for name in RELEVANT_FEATURE_NAMES:
        features[name] = 0.0
        features[f"{name}_missing"] = 1.0
    return features


def ensure_residue_entry(
    residue_features: Dict[ResidueKey, Dict[str, float]],
    key: ResidueKey,
) -> Dict[str, float]:
    if key not in residue_features:
        residue_features[key] = default_feature_dict()
    return residue_features[key]


def set_feature_value(
    residue_features: Dict[ResidueKey, Dict[str, float]],
    key: ResidueKey,
    name: str,
    value: float,
) -> None:
    entry = ensure_residue_entry(residue_features, key)
    entry[name] = float(value)
    entry[f"{name}_missing"] = 0.0


def iter_structure_dirs(root_dir: str | Path) -> Iterable[Path]:
    root = Path(root_dir)
    for job_dir in sorted(root.glob("job_*")):
        if not job_dir.is_dir():
            continue
        for structure_dir in sorted(job_dir.iterdir()):
            if structure_dir.is_dir():
                yield structure_dir


def parse_residue_order_from_pdb(pdb_path: Path) -> List[Tuple[str, str, int, str]]:
    residues: List[Tuple[str, str, int, str]] = []
    seen = set()

    with pdb_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith("#BEGIN_POSE_ENERGIES_TABLE"):
                break
            if not line.startswith("ATOM"):
                continue

            resname = line[17:20].strip()
            chain_id = line[21].strip()
            resseq_text = line[22:26].strip()
            icode = line[26].strip()

            if not resseq_text:
                continue

            key = (chain_id, int(resseq_text), icode)
            if key in seen:
                continue
            seen.add(key)
            residues.append((resname, chain_id, int(resseq_text), icode))

    return residues


def parse_pose_energy_rows(
    pdb_path: Path,
    residue_order: List[Tuple[str, str, int, str]],
) -> Dict[ResidueKey, Dict[str, float]]:
    rows: Dict[ResidueKey, Dict[str, float]] = {}
    header: Optional[List[str]] = None
    in_table = False

    with pdb_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()

            if line.startswith("#BEGIN_POSE_ENERGIES_TABLE"):
                in_table = True
                continue
            if line.startswith("#END_POSE_ENERGIES_TABLE"):
                break
            if not in_table or not line:
                continue

            parts = line.split()
            if not parts:
                continue

            if parts[0] == "label":
                header = parts
                continue
            if parts[0] in {"weights", "pose"} or header is None:
                continue

            label_match = ENERGY_LABEL_RE.match(parts[0])
            if label_match is None:
                continue

            pose_idx = int(label_match.group("pose_idx"))
            if pose_idx < 1 or pose_idx > len(residue_order):
                continue

            _resname, chain_id, resseq, icode = residue_order[pose_idx - 1]
            key = (chain_id, resseq, icode)
            row_lookup = dict(zip(header[1:], parts[1:]))

            row_features = rows.setdefault(key, {})
            for feature_name in ENERGY_TERM_NAMES:
                value = row_lookup.get(feature_name)
                if value is None:
                    continue
                row_features[feature_name] = float(value)

    return rows


def parse_residue_sasa(
    pdb_path: Path,
    residue_order: List[Tuple[str, str, int, str]],
) -> Dict[ResidueKey, float]:
    sasa_by_key: Dict[ResidueKey, float] = {}

    with pdb_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line.startswith("res_sasa_"):
                continue

            parts = line.split()
            if len(parts) != 2:
                continue

            pose_idx = int(parts[0][9:])
            if pose_idx < 1 or pose_idx > len(residue_order):
                continue

            _resname, chain_id, resseq, icode = residue_order[pose_idx - 1]
            sasa_by_key[(chain_id, resseq, icode)] = float(parts[1])

    return sasa_by_key


def parse_bsa_report(std_output_path: Path, chain_id: str) -> Dict[ResidueKey, float]:
    bsa_by_key: Dict[ResidueKey, float] = {}

    with std_output_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            match = BSA_REPORT_RE.search(raw_line)
            if match is None:
                continue

            resseq = int(match.group(2))
            icode = match.group(3).strip()
            value = float(match.group(4))
            bsa_by_key[(chain_id, resseq, icode)] = value

    return bsa_by_key


def parse_bluues_solv_energy(solv_path: Path) -> Dict[Tuple[str, int], float]:
    solv_energy: DefaultDict[Tuple[str, int], float] = defaultdict(float)

    with solv_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            parts = raw_line.split()
            if len(parts) < 8 or parts[0] != "SOLV" or parts[1] != "NRG":
                continue

            resname = parts[4]
            resseq = int(parts[5])
            value = float(parts[-1])
            solv_energy[(resname, resseq)] += value

    return dict(solv_energy)


def parse_bluues_pka(pka_path: Path) -> Dict[Tuple[str, int], Dict[str, float]]:
    pka_features: DefaultDict[Tuple[str, int], Dict[str, float]] = defaultdict(
        lambda: {
            "pKa_shift": 0.0,
            "dpKa_desolv": 0.0,
            "dpKa_bg": 0.0,
            "dpKa_titr": 0.0,
        }
    )

    with pka_path.open("r", encoding="utf-8", errors="ignore") as handle:
        next(handle, None)
        for raw_line in handle:
            parts = raw_line.split()
            if len(parts) < 10:
                continue

            resname = parts[1]
            resseq = int(parts[2])

            pka = float(parts[3])
            pka_0 = float(parts[4])
            dpka_self = float(parts[5])
            dpka_bg = float(parts[6])
            dpka_ii = float(parts[7])

            entry = pka_features[(resname, resseq)]
            entry["pKa_shift"] += pka_0 - pka
            entry["dpKa_desolv"] += dpka_self
            entry["dpKa_bg"] += dpka_bg
            entry["dpKa_titr"] += dpka_ii

    return dict(pka_features)


def build_residue_aliases(
    residue_order: List[Tuple[str, str, int, str]],
) -> Dict[Tuple[str, int], ResidueKey]:
    aliases: Dict[Tuple[str, int], ResidueKey] = {}
    ambiguous = set()

    for resname, chain_id, resseq, icode in residue_order:
        alias = (resname, resseq)
        key = (chain_id, resseq, icode)
        if alias in aliases and aliases[alias] != key:
            ambiguous.add(alias)
            continue
        aliases[alias] = key

    for alias in ambiguous:
        aliases.pop(alias, None)

    return aliases


def structure_dir_to_feature_lookup(structure_dir: str | Path) -> Dict[ResidueKey, Dict[str, float]]:
    structure_path = Path(structure_dir)
    structure_name = structure_path.name
    chain_id = infer_chain_id(structure_name)

    pdb_path = structure_path / f"{structure_name}.pdb"
    std_output_path = structure_path / "StdOutputScore.txt"
    pka_path = structure_path / f"{structure_name}_bluues.pka"
    solv_path = structure_path / f"{structure_name}_bluues.solv_nrg"

    if not pdb_path.is_file():
        raise FileNotFoundError(f"Missing structure PDB file: {pdb_path}")

    residue_order = parse_residue_order_from_pdb(pdb_path)
    residue_features = {
        (chain, resseq, icode): default_feature_dict()
        for _, chain, resseq, icode in residue_order
    }

    for key, feature_values in parse_pose_energy_rows(pdb_path, residue_order).items():
        for feature_name, value in feature_values.items():
            set_feature_value(residue_features, key, feature_name, value)

    for key, value in parse_residue_sasa(pdb_path, residue_order).items():
        set_feature_value(residue_features, key, "SASA", value)

    if std_output_path.is_file():
        for key, value in parse_bsa_report(std_output_path, chain_id).items():
            set_feature_value(residue_features, key, "BSA", value)

    residue_aliases = build_residue_aliases(residue_order)

    if solv_path.is_file():
        for alias, value in parse_bluues_solv_energy(solv_path).items():
            key = residue_aliases.get(alias)
            if key is not None:
                set_feature_value(residue_features, key, "SolvEnergy", value)

    if pka_path.is_file():
        for alias, feature_values in parse_bluues_pka(pka_path).items():
            key = residue_aliases.get(alias)
            if key is None:
                continue
            for feature_name, value in feature_values.items():
                set_feature_value(residue_features, key, feature_name, value)

    return residue_features
