#!/usr/bin/env python3
from __future__ import annotations

import re
import time
from pathlib import Path

import requests


# =========================
# EDIT THESE SETTINGS
# =========================
OUTPUT_DIR = Path("/home/mechti/Documents/uniprot_ec_from_pdb")

SLEEP_BETWEEN_REQUESTS = 0.2
REQUEST_TIMEOUT = 30
OVERWRITE_UNIPROT_TXT = False
# =========================


RCSB_ENTRY_URL = "https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
RCSB_POLYMER_ENTITY_URL = "https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_id}/{entity_id}"
UNIPROT_TXT_URL = "https://rest.uniprot.org/uniprotkb/{accession}.txt"


def safe_get(url: str, params: dict | None = None) -> requests.Response:
    response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response


def clean_field(value) -> str:
    if value is None:
        return ""
    return str(value).replace("\t", " ").replace("\n", " ").strip()


def extract_ec_numbers_from_text(text: str) -> list[str]:
    """
    Extract EC numbers from UniProt flat text.

    Matches forms like:
    EC=3.4.11.9
    EC=2.7.11.1
    EC=3.4.24.-
    EC=1.14.n.n
    """
    ec_numbers = re.findall(
        r"EC=([0-9n\-]+\.[0-9n\-]+\.[0-9n\-]+\.[0-9n\-]+)",
        text
    )

    seen = set()
    unique_ecs = []

    for ec in ec_numbers:
        if ec not in seen:
            seen.add(ec)
            unique_ecs.append(ec)

    return unique_ecs


def get_polymer_entity_ids_for_pdb(pdb_id: str) -> list[str]:
    pdb_id = pdb_id.upper().strip()

    url = RCSB_ENTRY_URL.format(pdb_id=pdb_id)
    response = safe_get(url)
    data = response.json()

    ids = data.get("rcsb_entry_container_identifiers", {}).get("polymer_entity_ids", [])
    return [str(x) for x in ids]


def get_uniprot_mappings_for_entity(pdb_id: str, entity_id: str) -> dict:
    """
    Return mapping details for one polymer entity.

    Example return:
    {
        "entity_id": "1",
        "description": "Hemoglobin subunit alpha",
        "chains": ["A", "C"],
        "uniprot_accessions": ["P69905"]
    }
    """
    pdb_id = pdb_id.upper().strip()
    entity_id = str(entity_id).strip()

    url = RCSB_POLYMER_ENTITY_URL.format(pdb_id=pdb_id, entity_id=entity_id)
    response = safe_get(url)
    data = response.json()

    description = data.get("rcsb_polymer_entity", {}).get("pdbx_description", "")

    id_block = data.get("rcsb_polymer_entity_container_identifiers", {}) or {}
    chains = id_block.get("auth_asym_ids", []) or []

    accessions = []
    seen = set()

    # Primary expected location
    refseqs = id_block.get("reference_sequence_identifiers", []) or []
    for ref in refseqs:
        if ref.get("database_name") == "UniProt":
            accession = str(ref.get("database_accession", "")).strip()
            if accession and accession not in seen:
                seen.add(accession)
                accessions.append(accession)

    # Fallback: scan recursively in case schema placement differs for some entries
    if not accessions:
        def walk(obj):
            if isinstance(obj, dict):
                db_name = obj.get("database_name")
                db_acc = obj.get("database_accession")
                if db_name == "UniProt" and db_acc:
                    yield str(db_acc).strip()

                for v in obj.values():
                    yield from walk(v)

            elif isinstance(obj, list):
                for item in obj:
                    yield from walk(item)

        for acc in walk(data):
            if acc and acc not in seen:
                seen.add(acc)
                accessions.append(acc)

    return {
        "entity_id": entity_id,
        "description": description,
        "chains": chains,
        "uniprot_accessions": accessions,
    }


def get_uniprot_mappings_for_pdb(pdb_id: str) -> list[dict]:
    pdb_id = pdb_id.upper().strip()

    entity_ids = get_polymer_entity_ids_for_pdb(pdb_id)

    mappings = []
    for entity_id in entity_ids:
        try:
            mapping = get_uniprot_mappings_for_entity(pdb_id, entity_id)
            mappings.append(mapping)
        except Exception as e:
            mappings.append({
                "entity_id": str(entity_id),
                "description": "",
                "chains": [],
                "uniprot_accessions": [],
                "error": str(e),
            })

        time.sleep(SLEEP_BETWEEN_REQUESTS)

    return mappings


def download_uniprot_txt(accession: str, txt_dir: Path) -> Path:
    accession = accession.strip()
    out_file = txt_dir / f"{accession}.txt"

    if out_file.exists() and not OVERWRITE_UNIPROT_TXT:
        return out_file

    url = UNIPROT_TXT_URL.format(accession=accession)
    response = safe_get(url)
    out_file.write_text(response.text, encoding="utf-8")

    return out_file


def process_pdb_ids(pdb_ids: list[str], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    txt_dir = output_dir / "uniprot_txt"
    txt_dir.mkdir(parents=True, exist_ok=True)

    out_tsv = output_dir / "pdb_to_uniprot_ec.tsv"

    with open(out_tsv, "w", encoding="utf-8") as out:
        out.write(
            "pdb_id\tentity_id\tchains\tdescription\tuniprot_accession\tec_numbers\tstatus\n"
        )

        for i, pdb_id in enumerate(pdb_ids, start=1):
            pdb_id = pdb_id.upper().strip()
            print(f"[{i}/{len(pdb_ids)}] Processing {pdb_id}")

            try:
                mappings = get_uniprot_mappings_for_pdb(pdb_id)
            except Exception as e:
                print(f"  Failed to get RCSB mapping for {pdb_id}: {e}")
                out.write(f"{pdb_id}\t\t\t\t\t\tRCSB_MAPPING_FAILED\n")
                time.sleep(SLEEP_BETWEEN_REQUESTS)
                continue

            if not mappings:
                print("  No polymer entities found")
                out.write(f"{pdb_id}\t\t\t\t\t\tNO_POLYMER_ENTITIES\n")
                time.sleep(SLEEP_BETWEEN_REQUESTS)
                continue

            for mapping in mappings:
                entity_id = clean_field(mapping.get("entity_id", ""))
                description = clean_field(mapping.get("description", ""))
                chains = clean_field(",".join(mapping.get("chains", [])))

                if "error" in mapping:
                    print(f"  Entity {entity_id}: failed to read entity details")
                    out.write(
                        f"{pdb_id}\t{entity_id}\t{chains}\t{description}\t\t\tENTITY_FETCH_FAILED\n"
                    )
                    continue

                accessions = mapping.get("uniprot_accessions", [])

                if not accessions:
                    print(f"  Entity {entity_id} ({chains or 'no chains'}): no UniProt found")
                    out.write(
                        f"{pdb_id}\t{entity_id}\t{chains}\t{description}\t\t\tNO_UNIPROT_FOUND\n"
                    )
                    continue

                for accession in accessions:
                    try:
                        txt_file = download_uniprot_txt(accession, txt_dir)
                        text = txt_file.read_text(encoding="utf-8")
                        ec_numbers = extract_ec_numbers_from_text(text)
                        ec_field = ";".join(ec_numbers)

                        out.write(
                            f"{pdb_id}\t{entity_id}\t{chains}\t{description}\t{accession}\t{ec_field}\tOK\n"
                        )

                        if ec_numbers:
                            print(
                                f"  Entity {entity_id} ({chains}) -> {accession} -> EC: {', '.join(ec_numbers)}"
                            )
                        else:
                            print(
                                f"  Entity {entity_id} ({chains}) -> {accession} -> no EC found"
                            )

                    except Exception as e:
                        print(f"  Failed for accession {accession}: {e}")
                        out.write(
                            f"{pdb_id}\t{entity_id}\t{chains}\t{description}\t{accession}\t\tDOWNLOAD_OR_PARSE_FAILED\n"
                        )

                    time.sleep(SLEEP_BETWEEN_REQUESTS)

    return out_tsv


def read_pdb_ids_from_file(file_path: Path) -> list[str]:
    pdb_ids = []
    seen = set()

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            match = re.search(r"\b([A-Za-z0-9]{4})\b", line)
            if not match:
                continue

            pdb_id = match.group(1).upper()

            if pdb_id not in seen:
                seen.add(pdb_id)
                pdb_ids.append(pdb_id)

    return pdb_ids


if __name__ == "__main__":
    """
    Built-in test case.
    Uses 4HHB by default.
    """

    test_pdb_ids = [
        "4HHB",
        "1B0F",
        "1AR1"
    ]

    print("Running built-in test case")
    print("PDB IDs:", ", ".join(test_pdb_ids))
    print()

    out_tsv = process_pdb_ids(test_pdb_ids, OUTPUT_DIR)

    print()
    print("Done.")
    print(f"Summary file: {out_tsv}")
    print(f"UniProt text files folder: {OUTPUT_DIR / 'uniprot_txt'}")