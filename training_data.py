from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Optional, Tuple

from data_structures import PocketRecord
from graph_construction import (
    canonical_ring_edges_output_path,
    extract_metal_pockets_from_structure,
    parse_structure_file,
)
from project_paths import (
    CATALYTIC_ONLY_SUMMARY_CSV,
    MAHOMES_TRAIN_SET_DIR,
)
from training_labels import (
    infer_metal_target_class_from_pocket,
    normalize_ec_numbers,
    parse_ec_top_level_from_structure_path,
    parse_structure_identity,
)


DEFAULT_STRUCTURE_DIR = MAHOMES_TRAIN_SET_DIR
DEFAULT_TRAIN_SUMMARY_CSV = CATALYTIC_ONLY_SUMMARY_CSV
SUMMARY_REQUIRED_COLUMNS = frozenset({"pdbid", "metal residue number", "EC number"})

SiteKey = Tuple[str, str, str]


def _validate_summary_columns(fieldnames: Optional[List[str]], summary_csv: Path) -> None:
    if fieldnames is None:
        raise ValueError(f"Could not read CSV header from: {summary_csv}")

    missing_columns = SUMMARY_REQUIRED_COLUMNS.difference(fieldnames)
    if missing_columns:
        raise ValueError(f"Missing required columns {sorted(missing_columns)} in {summary_csv}")


def load_allowed_site_keys(summary_csv: Path) -> set[SiteKey]:
    with summary_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        _validate_summary_columns(reader.fieldnames, summary_csv)

        keys = set()
        for row in reader:
            pdbid = row["pdbid"].strip().lower()
            ec_number = normalize_ec_numbers(row["EC number"])
            metal_residue_number = row["metal residue number"].strip()
            if not pdbid or not ec_number or not metal_residue_number:
                continue
            keys.add((pdbid, ec_number, metal_residue_number))
    return keys


def resolve_allowed_site_keys(summary_csv: Optional[Path]) -> Optional[set[SiteKey]]:
    if summary_csv is None:
        return None

    summary_csv = Path(summary_csv)
    if not summary_csv.exists():
        raise FileNotFoundError(f"Training summary file not found: {summary_csv}")
    return load_allowed_site_keys(summary_csv)


def pocket_matches_allowed_sites(
    pocket: PocketRecord,
    structure_path: Path,
    allowed_site_keys: set[SiteKey],
) -> bool:
    pdbid, _, ec_number = parse_structure_identity(structure_path.stem)
    metal_site_ids = pocket.metadata.get("metal_site_ids", [])
    if not isinstance(metal_site_ids, list):
        return False

    for site_id in metal_site_ids:
        if not isinstance(site_id, tuple) or len(site_id) != 3:
            continue
        chain_id, resseq, _icode = site_id
        try:
            normalized_resseq = int(str(resseq).strip())
        except (TypeError, ValueError):
            continue
        site_key = (pdbid, ec_number, f"{str(chain_id).strip()}_{normalized_resseq}")
        if site_key in allowed_site_keys:
            return True
    return False


def assign_supervision_labels(pocket: PocketRecord, ec_label: Optional[int]) -> None:
    pocket.y_metal = infer_metal_target_class_from_pocket(pocket)
    if ec_label is not None:
        pocket.y_ec = ec_label


def pocket_has_full_supervision(pocket: PocketRecord) -> bool:
    return pocket.y_metal is not None and pocket.y_ec is not None


def find_structure_files(structure_dir: Path) -> List[Path]:
    patterns = ("*.pdb", "*.cif", "*.mmcif")
    files: List[Path] = []
    for pattern in patterns:
        files.extend(structure_dir.rglob(pattern))
    return sorted(path for path in files if path.is_file())


def load_labeled_pockets_from_dir(
    structure_dir: Path,
    max_cases: Optional[int] = None,
    require_full_labels: bool = True,
    summary_csv: Optional[Path] = DEFAULT_TRAIN_SUMMARY_CSV,
) -> List[PocketRecord]:
    structure_files = find_structure_files(structure_dir)
    if not structure_files:
        raise FileNotFoundError(f"No structure files found under {structure_dir}")

    allowed_site_keys = resolve_allowed_site_keys(summary_csv)

    pockets: List[PocketRecord] = []
    for structure_path in structure_files:
        structure = parse_structure_file(str(structure_path), structure_id=structure_path.stem)
        extracted = extract_metal_pockets_from_structure(structure, structure_id=structure_path.stem)
        if not extracted:
            continue

        ec_label = parse_ec_top_level_from_structure_path(structure_path)

        for pocket in extracted:
            if allowed_site_keys is not None and not pocket_matches_allowed_sites(
                pocket,
                structure_path,
                allowed_site_keys,
            ):
                continue
            pocket.metadata["source_path"] = str(structure_path)
            pocket.metadata.setdefault(
                "ring_edges_expected_path",
                str(canonical_ring_edges_output_path(structure_path)),
            )
            assign_supervision_labels(pocket, ec_label)
            if require_full_labels and not pocket_has_full_supervision(pocket):
                continue
            pockets.append(pocket)
            if max_cases is not None and len(pockets) >= max_cases:
                return pockets

    if not pockets:
        if require_full_labels:
            raise ValueError(
                f"No fully labeled metal-centered pockets were extracted from {structure_dir}"
            )
        raise ValueError(f"No metal-centered pockets were extracted from {structure_dir}")
    return pockets


def load_training_pockets_from_dir(
    structure_dir: Path,
    require_full_labels: bool = True,
    summary_csv: Optional[Path] = DEFAULT_TRAIN_SUMMARY_CSV,
) -> List[PocketRecord]:
    return load_labeled_pockets_from_dir(
        structure_dir=structure_dir,
        max_cases=None,
        require_full_labels=require_full_labels,
        summary_csv=summary_csv,
    )


def load_smoke_test_pockets_from_dir(
    structure_dir: Path,
    max_cases: int = 4,
    require_full_labels: bool = True,
    summary_csv: Optional[Path] = DEFAULT_TRAIN_SUMMARY_CSV,
) -> List[PocketRecord]:
    return load_labeled_pockets_from_dir(
        structure_dir=structure_dir,
        max_cases=max_cases,
        require_full_labels=require_full_labels,
        summary_csv=summary_csv,
    )
