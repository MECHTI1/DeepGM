#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import urllib.error
import urllib.request
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PREP_DIR = SCRIPT_DIR.parent

DEFAULT_INPUT_CSV = PREP_DIR / "pinmymetal_files" / "classmodel_train_set"
DEFAULT_OUTPUT_DIR = Path("/media/Data/pinmymetal_sets/train/af2")

RCSB_ENTRY_URL = "https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
RCSB_POLYMER_ENTITY_URL = "https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_id}/{entity_id}"
AF2_API_URL = "https://alphafold.ebi.ac.uk/api/prediction/{accession}"
TIMEOUT = 45


def get_json(url: str):
    with urllib.request.urlopen(url, timeout=TIMEOUT) as response:
        return json.loads(response.read().decode("utf-8"))


def clean_pdb_id(raw: str) -> str | None:
    value = str(raw).strip().lower()
    if len(value) == 4 and value.isalnum():
        return value
    return None


def read_unique_pdb_ids(csv_path: Path) -> list[str]:
    seen: set[str] = set()
    pdb_ids: list[str] = []

    with open(csv_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Missing header in {csv_path}")

        pdb_col = next((c for c in reader.fieldnames if c.strip().lower() == "pdbid"), None)
        if pdb_col is None:
            raise ValueError(f"'pdbid' column not found in {csv_path}")

        for row_num, row in enumerate(reader, start=2):
            pdb_id = clean_pdb_id(row.get(pdb_col, ""))
            if pdb_id is None:
                print(f"[warning] row {row_num}: invalid pdbid -> skipped")
                continue
            if pdb_id not in seen:
                seen.add(pdb_id)
                pdb_ids.append(pdb_id)

    return pdb_ids


def find_uniprot_mappings(pdb_id: str) -> list[tuple[str, str]]:
    entry = get_json(RCSB_ENTRY_URL.format(pdb_id=pdb_id.upper()))
    entity_ids = entry.get("rcsb_entry_container_identifiers", {}).get("polymer_entity_ids", [])

    mappings: set[tuple[str, str]] = set()
    for entity_id in entity_ids:
        entity_id = str(entity_id)
        entity = get_json(
            RCSB_POLYMER_ENTITY_URL.format(pdb_id=pdb_id.upper(), entity_id=entity_id)
        )
        container_ids = entity.get("rcsb_polymer_entity_container_identifiers", {}) or {}
        refs = container_ids.get("reference_sequence_identifiers", []) or []

        for ref in refs:
            if ref.get("database_name") == "UniProt":
                acc = str(ref.get("database_accession", "")).strip()
                if acc:
                    mappings.add((entity_id, acc))

        # Fallback used by some entries
        for acc in container_ids.get("uniprot_ids", []) or []:
            acc = str(acc).strip()
            if acc:
                mappings.add((entity_id, acc))

        # Extra fallback from alignment block
        for aln in entity.get("rcsb_polymer_entity_align", []) or []:
            if aln.get("reference_database_name") == "UniProt":
                acc = str(aln.get("reference_database_accession", "")).strip()
                if acc:
                    mappings.add((entity_id, acc))

    return sorted(mappings)


def find_af2_cif_url(accession: str) -> str | None:
    records = get_json(AF2_API_URL.format(accession=accession))
    if isinstance(records, list) and records:
        return records[0].get("cifUrl")
    return None


def download(url: str, out_file: Path, overwrite: bool) -> str:
    if out_file.exists() and not overwrite:
        return "SKIPPED"
    with urllib.request.urlopen(url, timeout=TIMEOUT) as response:
        out_file.write_bytes(response.read())
    return "OK"


def run(input_csv: Path, output_dir: Path, limit: int | None, overwrite: bool) -> Path:
    pdb_ids = read_unique_pdb_ids(input_csv)
    if limit is not None:
        pdb_ids = pdb_ids[:limit]

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "download_summary.tsv"
    mapping_path = output_dir / "pdb_entity_uniprot.tsv"

    with (
        open(summary_path, "w", encoding="utf-8") as summary,
        open(mapping_path, "w", encoding="utf-8") as mapping_file,
    ):
        summary.write("pdb_id\tentity_id\tuniprot_accession\tstatus\tfile\tnote\n")
        mapping_file.write("pdb_id\tentity_id\t_struct_ref.pdbx_db_accession\n")

        for idx, pdb_id in enumerate(pdb_ids, start=1):
            print(f"[{idx}/{len(pdb_ids)}] {pdb_id.upper()}")
            pdb_dir = output_dir / pdb_id
            pdb_dir.mkdir(parents=True, exist_ok=True)

            try:
                mappings = find_uniprot_mappings(pdb_id)
            except Exception as exc:
                summary.write(f"{pdb_id}\t\t\tRCSB_FAILED\t\t{str(exc).replace(chr(9), ' ')}\n")
                print(f"  [failed] RCSB mapping: {exc}")
                continue

            if not mappings:
                summary.write(f"{pdb_id}\t\t\tNO_UNIPROT\t\t\n")
                print("  [info] no UniProt mapping")
                continue

            for entity_id, accession in mappings:
                mapping_file.write(f"{pdb_id}\t{entity_id}\t{accession}\n")
                try:
                    cif_url = find_af2_cif_url(accession)
                except urllib.error.HTTPError as exc:
                    if exc.code == 404:
                        summary.write(f"{pdb_id}\t{entity_id}\t{accession}\tNO_AF2\t\t404\n")
                        print(f"  entity {entity_id} | {accession}: no AF2")
                    else:
                        summary.write(
                            f"{pdb_id}\t{entity_id}\t{accession}\tAF2_API_FAILED\t\tHTTP {exc.code}\n"
                        )
                        print(f"  entity {entity_id} | {accession}: AF2 API HTTP {exc.code}")
                    continue
                except Exception as exc:
                    summary.write(
                        f"{pdb_id}\t{entity_id}\t{accession}\tAF2_API_FAILED\t\t{str(exc).replace(chr(9), ' ')}\n"
                    )
                    print(f"  entity {entity_id} | {accession}: AF2 API failed")
                    continue

                if not cif_url:
                    summary.write(f"{pdb_id}\t{entity_id}\t{accession}\tNO_CIF_URL\t\t\n")
                    print(f"  entity {entity_id} | {accession}: no cifUrl")
                    continue

                out_dir = pdb_dir / accession
                out_dir.mkdir(parents=True, exist_ok=True)
                out_file = out_dir / Path(cif_url).name

                try:
                    status = download(cif_url, out_file, overwrite=overwrite)
                    summary.write(f"{pdb_id}\t{entity_id}\t{accession}\t{status}\t{out_file.name}\t\n")
                    print(f"  entity {entity_id} | {accession}: {status}")
                except Exception as exc:
                    summary.write(
                        f"{pdb_id}\t{entity_id}\t{accession}\tDOWNLOAD_FAILED\t\t{str(exc).replace(chr(9), ' ')}\n"
                    )
                    print(f"  entity {entity_id} | {accession}: download failed")

    return summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read classmodel_train_set pdbid and download matching AF2 CIF structures."
    )
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=None, help="Process only first N unique PDB IDs")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input_csv.exists():
        raise FileNotFoundError(f"Input file not found: {args.input_csv}")

    print(f"Input: {args.input_csv}")
    print(f"Output: {args.output_dir}")
    print("")
    summary_path = run(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        limit=args.limit,
        overwrite=args.overwrite,
    )
    print("")
    print("Done.")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
