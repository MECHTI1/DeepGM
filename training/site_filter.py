from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Optional, Tuple

from data_structures import PocketRecord
from training.labels import normalize_ec_numbers, parse_structure_identity

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
            if pdbid and ec_number and metal_residue_number:
                keys.add((pdbid, ec_number, metal_residue_number))
    return keys


def resolve_allowed_site_keys(summary_csv: Path | None) -> set[SiteKey] | None:
    if summary_csv is None:
        return None

    summary_path = Path(summary_csv)
    if not summary_path.exists():
        raise FileNotFoundError(f"Training summary file not found: {summary_path}")
    return load_allowed_site_keys(summary_path)


def pocket_matches_allowed_sites(
    pocket: PocketRecord,
    structure_path: Path,
    allowed_site_keys: set[SiteKey],
) -> bool:
    pdbid, _chain_id, ec_number = parse_structure_identity(structure_path.stem)
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
        if (pdbid, ec_number, f"{str(chain_id).strip()}_{normalized_resseq}") in allowed_site_keys:
            return True
    return False


__all__ = [
    "SUMMARY_REQUIRED_COLUMNS",
    "SiteKey",
    "load_allowed_site_keys",
    "pocket_matches_allowed_sites",
    "resolve_allowed_site_keys",
]
