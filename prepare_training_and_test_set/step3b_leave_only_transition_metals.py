#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path


JOB_ROOT = Path("/media/Data/pinmymetal_sets/mahomes/train_set")
SUMMARY_DIR = JOB_ROOT / "data_summarizing_tables"
INPUT_CSV = SUMMARY_DIR / "data_summarazing_table.csv"
OUTPUT_CSV = SUMMARY_DIR / "data_summarazing_table_transition_metals.csv"

# Keep this aligned with the earlier transition-metal filter in the pipeline.
TRANSITION_METALS = {"MN", "FE", "CO", "NI", "CU", "ZN"}
METAL_TYPE_COLUMN = "metal residue type"


def is_transition_metal(value: str) -> bool:
    return value.strip().upper() in TRANSITION_METALS


def main() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Input summary not found: {INPUT_CSV}. Run step3_concat_mahomes_and_ec.py first."
        )

    with INPUT_CSV.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Could not read CSV header from: {INPUT_CSV}")
        if METAL_TYPE_COLUMN not in reader.fieldnames:
            raise ValueError(
                f"Expected column '{METAL_TYPE_COLUMN}' in {INPUT_CSV}, got: {reader.fieldnames}"
            )

        rows = list(reader)

    filtered_rows = [row for row in rows if is_transition_metal(row.get(METAL_TYPE_COLUMN, ""))]
    if not filtered_rows:
        raise ValueError(f"No transition-metal rows found in: {INPUT_CSV}")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=reader.fieldnames)
        writer.writeheader()
        writer.writerows(filtered_rows)

    print(f"Read {len(rows)} rows from {INPUT_CSV}")
    print(f"Wrote {len(filtered_rows)} transition-metal rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
