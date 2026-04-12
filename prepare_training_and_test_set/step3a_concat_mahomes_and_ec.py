#!/usr/bin/env python3
from __future__ import annotations

import csv
import re

from project_paths import MAHOMES_SUMMARY_DIR, MAHOMES_TRAIN_SET_DIR
from training.labels import normalize_ec_numbers

JOB_ROOT = MAHOMES_TRAIN_SET_DIR
SUMMARY_DIR = MAHOMES_SUMMARY_DIR
OUTPUT_CSV = SUMMARY_DIR / "data_summarazing_table.csv"

STRUCTURE_CHAIN_RE = re.compile(r"__chain_([^_]+)")
STRUCTURE_EC_RE = re.compile(r"__EC_([^_]+)")
SITE_DIR_RE = re.compile(r"^(?P<metal_type>[A-Za-z0-9]+)_(?P<metal_resseq>\d+)__.+$")


def parse_pdbid(structure_dir_name: str) -> str:
    return structure_dir_name.split("__", 1)[0].strip()


def parse_ec_numbers(structure_dir_name: str) -> str:
    return normalize_ec_numbers(";".join(STRUCTURE_EC_RE.findall(structure_dir_name)))


def parse_chain_id(structure_dir_name: str) -> str:
    match = STRUCTURE_CHAIN_RE.search(structure_dir_name)
    if match is None:
        raise ValueError(f"Could not parse chain id from structure directory name: {structure_dir_name}")
    return match.group(1).strip()


def parse_site_dir(site_dir_name: str) -> tuple[str, int] | None:
    match = SITE_DIR_RE.match(site_dir_name)
    if match is None:
        return None
    metal_type = match.group("metal_type")
    metal_resseq = int(match.group("metal_resseq"))
    return metal_type, metal_resseq


def iter_summary_rows(job_root: Path):
    for job_dir in sorted(job_root.glob("job_*")):
        if not job_dir.is_dir():
            continue
        for structure_dir in sorted(job_dir.iterdir()):
            if not structure_dir.is_dir():
                continue

            pdbid = parse_pdbid(structure_dir.name)
            chain_id = parse_chain_id(structure_dir.name)
            ec_number = parse_ec_numbers(structure_dir.name)

            for site_dir in sorted(structure_dir.iterdir()):
                if not site_dir.is_dir():
                    continue

                parsed = parse_site_dir(site_dir.name)
                if parsed is None:
                    continue

                metal_type, metal_resseq = parsed
                yield {
                    "pdbid": pdbid,
                    "metal residue number": f"{chain_id}_{metal_resseq}",
                    "metal residue type": metal_type,
                    "EC number": ec_number,
                }


def main() -> None:
    if not JOB_ROOT.exists():
        raise FileNotFoundError(f"Job root not found: {JOB_ROOT}")

    rows = list(iter_summary_rows(JOB_ROOT))
    if not rows:
        raise ValueError(f"No metal-site rows were found under: {JOB_ROOT}")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["pdbid", "metal residue number", "metal residue type", "EC number"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
