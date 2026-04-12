#!/usr/bin/env python3
from __future__ import annotations

import csv
import re

from project_paths import (
    CATALYTIC_ONLY_SUMMARY_CSV,
    MAHOMES_TRAIN_SET_DIR,
    PREDICTION_RESULTS_SUMMARY_CSV,
    TRANSITION_METALS_SUMMARY_CSV,
    WHETHER_CATALYTIC_SUMMARY_CSV,
)
from training_labels import normalize_ec_numbers


JOB_ROOT = MAHOMES_TRAIN_SET_DIR
INPUT_SUMMARY_CSV = TRANSITION_METALS_SUMMARY_CSV
PREDICTION_SUMMARY_CSV = PREDICTION_RESULTS_SUMMARY_CSV
OUTPUT_CSV = WHETHER_CATALYTIC_SUMMARY_CSV
CATALYTIC_ONLY_CSV = CATALYTIC_ONLY_SUMMARY_CSV

PREDICTION_INPUT_RE = re.compile(r"^(?P<pdbid>[^_]+)__chain_(?P<chain>[^_]+)__EC_(?P<ec>.+)$")
CATALYTIC_COLUMN = "whether_catalytic"


def parse_prediction_label(value: str) -> int:
    return 1 if value.strip().lower() == "catalytic" else 0


def parse_prediction_input_file(value: str) -> tuple[str, str, str]:
    match = PREDICTION_INPUT_RE.match(value.strip())
    if match is None:
        raise ValueError(f"Could not parse prediction input file: {value!r}")
    return (
        match.group("pdbid").strip().lower(),
        match.group("chain").strip(),
        normalize_ec_numbers(match.group("ec")),
    )


def format_chain_residue_number(chain_id: str, residue_number: int) -> str:
    return f"{chain_id}_{residue_number}"


def iter_prediction_summary_rows(job_root: Path):
    for pred_path in sorted(job_root.glob("job_*/predictions.csv")):
        job_name = pred_path.parent.name
        with pred_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"Could not read CSV header from: {pred_path}")

            for row in reader:
                pdbid, chain_id, ec_number = parse_prediction_input_file(row["input file"])
                catalytic_value = parse_prediction_label(row["prediction"])

                for idx in range(1, 5):
                    metal_type = (row.get(f"Name{idx}") or "").strip()
                    residue_number = (row.get(f"Res#{idx}") or "").strip()
                    if not metal_type or not residue_number:
                        continue

                    yield {
                        "job_name": job_name,
                        "jobid": row["jobid"].strip(),
                        "input file": row["input file"].strip(),
                        "site#": row["site#"].strip(),
                        "pdbid": pdbid,
                        "chain": chain_id,
                        "EC number": ec_number,
                        "metal residue type prediction": metal_type,
                        "metal residue number": format_chain_residue_number(
                            chain_id,
                            int(float(residue_number)),
                        ),
                        "percent catalytic predictions": row["percent catalytic predictions"].strip(),
                        "prediction": row["prediction"].strip(),
                        CATALYTIC_COLUMN: catalytic_value,
                    }


def write_prediction_summary(job_root: Path) -> list[dict[str, object]]:
    rows = list(iter_prediction_summary_rows(job_root))
    if not rows:
        raise ValueError(f"No prediction rows found under: {job_root}")

    fieldnames = [
        "job_name",
        "jobid",
        "input file",
        "site#",
        "pdbid",
        "chain",
        "EC number",
        "metal residue type prediction",
        "metal residue number",
        "percent catalytic predictions",
        "prediction",
        CATALYTIC_COLUMN,
    ]

    with PREDICTION_SUMMARY_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return rows


def build_catalytic_lookup(prediction_rows: list[dict[str, object]]) -> tuple[dict[tuple[str, str, str], int], int]:
    lookup: dict[tuple[str, str, str], int] = {}
    conflict_count = 0

    for row in prediction_rows:
        key = (
            str(row["pdbid"]).strip().lower(),
            str(row["EC number"]).strip(),
            str(row["metal residue number"]).strip(),
        )
        value = int(row[CATALYTIC_COLUMN])

        if key in lookup and lookup[key] != value:
            conflict_count += 1
            lookup[key] = max(lookup[key], value)
            continue

        lookup[key] = value

    return lookup, conflict_count


def main() -> None:
    if not JOB_ROOT.exists():
        raise FileNotFoundError(f"Job root not found: {JOB_ROOT}")
    if not INPUT_SUMMARY_CSV.exists():
        raise FileNotFoundError(
            f"Input summary not found: {INPUT_SUMMARY_CSV}. Run step3b_leave_only_transition_metals.py first."
        )

    prediction_rows = write_prediction_summary(JOB_ROOT)
    catalytic_lookup, conflict_count = build_catalytic_lookup(prediction_rows)

    with INPUT_SUMMARY_CSV.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Could not read CSV header from: {INPUT_SUMMARY_CSV}")

        output_fieldnames = list(reader.fieldnames)
        if CATALYTIC_COLUMN not in output_fieldnames:
            output_fieldnames.append(CATALYTIC_COLUMN)

        output_rows = []
        matched_rows = 0
        for row in reader:
            key = (
                row["pdbid"].strip().lower(),
                row["EC number"].strip(),
                row["metal residue number"].strip(),
            )
            catalytic_value = catalytic_lookup.get(key, 0)
            if key in catalytic_lookup:
                matched_rows += 1
            row[CATALYTIC_COLUMN] = catalytic_value
            output_rows.append(row)

    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    catalytic_only_fieldnames = [name for name in output_fieldnames if name != CATALYTIC_COLUMN]
    catalytic_only_rows = [
        {key: value for key, value in row.items() if key != CATALYTIC_COLUMN}
        for row in output_rows
        if int(row[CATALYTIC_COLUMN]) == 1
    ]
    with CATALYTIC_ONLY_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=catalytic_only_fieldnames)
        writer.writeheader()
        writer.writerows(catalytic_only_rows)

    print(f"Wrote {len(prediction_rows)} rows to {PREDICTION_SUMMARY_CSV}")
    print(f"Read {len(output_rows)} rows from {INPUT_SUMMARY_CSV}")
    print(f"Matched {matched_rows} rows to prediction residues")
    print(f"Resolved {conflict_count} conflicting lookup entries by preferring catalytic (1)")
    print(f"Wrote {len(output_rows)} rows to {OUTPUT_CSV}")
    print(f"Wrote {len(catalytic_only_rows)} catalytic rows to {CATALYTIC_ONLY_CSV}")


if __name__ == "__main__":
    main()
