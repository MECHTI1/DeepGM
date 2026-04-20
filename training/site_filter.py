from __future__ import annotations

import csv
from collections.abc import Collection, Mapping
from pathlib import Path
from typing import List, Optional, Tuple

from data_structures import PocketRecord
from training.labels import normalize_ec_number_list, parse_structure_identity

SUMMARY_REQUIRED_COLUMNS = frozenset({"pdbid", "metal residue number", "EC number"})
METAL_RESIDUE_TYPE_COLUMN = "metal residue type"
SiteKey = Tuple[str, str, str]
AllowedSiteMetalLabels = dict[SiteKey, str]


def _validate_summary_columns(fieldnames: Optional[List[str]], summary_csv: Path) -> None:
    if fieldnames is None:
        raise ValueError(f"Could not read CSV header from: {summary_csv}")

    missing_columns = SUMMARY_REQUIRED_COLUMNS.difference(fieldnames)
    if missing_columns:
        raise ValueError(f"Missing required columns {sorted(missing_columns)} in {summary_csv}")


def _iter_normalized_summary_rows(summary_csv: Path) -> tuple[list[tuple[SiteKey, dict[str, str]]], list[str]]:
    with summary_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        _validate_summary_columns(reader.fieldnames, summary_csv)

        normalized_rows: list[tuple[SiteKey, dict[str, str]]] = []
        for row in reader:
            pdbid = row["pdbid"].strip().lower()
            ec_number = normalize_ec_number_list(row["EC number"])
            metal_residue_number = row["metal residue number"].strip()
            if pdbid and ec_number and metal_residue_number:
                normalized_rows.append(((pdbid, ec_number, metal_residue_number), row))
    return normalized_rows, list(reader.fieldnames or [])


def load_allowed_site_metal_labels(summary_csv: Path) -> AllowedSiteMetalLabels:
    normalized_rows, fieldnames = _iter_normalized_summary_rows(summary_csv)
    if METAL_RESIDUE_TYPE_COLUMN not in fieldnames:
        raise ValueError(f"Missing required columns ['{METAL_RESIDUE_TYPE_COLUMN}'] in {summary_csv}")

    metal_labels: AllowedSiteMetalLabels = {}
    for site_key, row in normalized_rows:
        metal_residue_type = row.get(METAL_RESIDUE_TYPE_COLUMN, "").strip().upper()
        if metal_residue_type:
            metal_labels[site_key] = metal_residue_type
    return metal_labels


def resolve_allowed_site_metal_labels(summary_csv: Path | None) -> AllowedSiteMetalLabels | None:
    if summary_csv is None:
        return None

    summary_path = Path(summary_csv)
    if not summary_path.exists():
        raise FileNotFoundError(f"Training summary file not found: {summary_path}")
    return load_allowed_site_metal_labels(summary_path)


def matched_site_keys_for_pocket(
    pocket: PocketRecord,
    structure_path: Path,
    allowed_site_keys: Collection[SiteKey],
) -> set[SiteKey]:
    matched_site_keys: set[SiteKey] = set()
    pdbid, _chain_id, ec_number = parse_structure_identity(structure_path.stem)
    metal_site_ids = pocket.metadata.get("metal_site_ids", [])
    if not isinstance(metal_site_ids, list):
        return matched_site_keys

    for site_id in metal_site_ids:
        if not isinstance(site_id, tuple) or len(site_id) != 3:
            continue
        chain_id, resseq, _icode = site_id
        try:
            normalized_resseq = int(str(resseq).strip())
        except (TypeError, ValueError):
            continue
        candidate_key = (pdbid, ec_number, f"{str(chain_id).strip()}_{normalized_resseq}")
        if candidate_key in allowed_site_keys:
            matched_site_keys.add(candidate_key)
    return matched_site_keys


def pocket_matches_allowed_sites(
    pocket: PocketRecord,
    structure_path: Path,
    allowed_site_keys: Collection[SiteKey],
) -> bool:
    return bool(matched_site_keys_for_pocket(pocket, structure_path, allowed_site_keys))


def matched_site_metal_types(
    pocket: PocketRecord,
    structure_path: Path,
    allowed_site_metal_labels: Mapping[SiteKey, str],
) -> set[str]:
    return {
        allowed_site_metal_labels[site_key]
        for site_key in matched_site_keys_for_pocket(pocket, structure_path, allowed_site_metal_labels)
    }
