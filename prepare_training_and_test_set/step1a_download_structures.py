from pathlib import Path
import csv
import re
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# =========================
# EDIT THESE SETTINGS
# =========================
TRAIN_FILE = "pinmymetal_files/classmodel_train_set"
TEST_FILE = "pinmymetal_files/classmodel_test_set"

BASE_OUTPUT_DIR = "/media/Data/pinmymetal sets"

OVERWRITE = False
MAX_WORKERS = 2
# =========================

print_lock = threading.Lock()


def safe_print(*args, **kwargs):
    with print_lock:
        print(*args, **kwargs)


def clean_pdb_id(value):
    value = str(value).strip().lower()
    if not value:
        return None

    if re.fullmatch(r"[A-Za-z0-9]{4}", value):
        return value

    return None


def read_ids_from_csv(path):
    ids = []
    seen = set()

    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise ValueError(f"Could not read CSV header from: {path}")

        pdbid_col = None
        for col in reader.fieldnames:
            if col.strip().lower() == "pdbid":
                pdbid_col = col
                break

        if pdbid_col is None:
            raise ValueError(f"Could not find a 'pdbid' column in: {path}")

        for row_num, row in enumerate(reader, start=2):
            pdb_id = clean_pdb_id(row.get(pdbid_col))

            if pdb_id is None:
                safe_print(f"[warning] {path} row {row_num}: invalid pdbid -> skipped")
                continue

            if pdb_id not in seen:
                seen.add(pdb_id)
                ids.append(pdb_id)

    return ids


def download_file(url, out_path, overwrite=False):
    if out_path.exists() and not overwrite:
        return ("skipped", out_path.name, "already exists")

    try:
        urllib.request.urlretrieve(url, out_path)
        return ("downloaded", out_path.name, None)

    except urllib.error.HTTPError as e:
        return ("failed", out_path.name, f"HTTP error: {e.code}")

    except urllib.error.URLError as e:
        return ("failed", out_path.name, f"URL error: {e.reason}")

    except Exception as e:
        return ("failed", out_path.name, f"unexpected error: {e}")


def prepare_download_jobs(pdb_ids, pdb_dir, cif_dir):
    jobs = []

    for pdb_id in pdb_ids:
        pdb_id_upper = pdb_id.upper()

        pdb_url = f"https://files.rcsb.org/download/{pdb_id_upper}.pdb"
        cif_url = f"https://files.rcsb.org/download/{pdb_id_upper}.cif"

        pdb_out = pdb_dir / f"{pdb_id}.pdb"
        cif_out = cif_dir / f"{pdb_id}.cif"

        jobs.append((pdb_url, pdb_out))
        jobs.append((cif_url, cif_out))

    return jobs


def download_set(input_csv, pdb_dir, cif_dir, overwrite=False, max_workers=2):
    pdb_ids = read_ids_from_csv(input_csv)

    if not pdb_ids:
        safe_print(f"No valid PDB IDs found in {input_csv}")
        return

    safe_print(f"\nReading from: {input_csv}")
    safe_print(f"Found {len(pdb_ids)} unique PDB IDs")
    safe_print(f"PDB output: {pdb_dir}")
    safe_print(f"CIF output: {cif_dir}\n")

    jobs = prepare_download_jobs(pdb_ids, pdb_dir, cif_dir)
    total_jobs = len(jobs)

    downloaded = 0
    skipped = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(download_file, url, out_path, overwrite)
            for url, out_path in jobs
        ]

        for i, future in enumerate(as_completed(futures), start=1):
            status, name, message = future.result()

            if status == "downloaded":
                downloaded += 1
                safe_print(f"[{i}/{total_jobs}] [ok] {name}")

            elif status == "skipped":
                skipped += 1
                safe_print(f"[{i}/{total_jobs}] [skip] {name} ({message})")

            else:
                failed += 1
                safe_print(f"[{i}/{total_jobs}] [failed] {name} ({message})")

    safe_print("\nFinished set:")
    safe_print(f"Downloaded files: {downloaded}")
    safe_print(f"Skipped files:    {skipped}")
    safe_print(f"Failed files:     {failed}")


def main():
    train_file = Path(TRAIN_FILE)
    test_file = Path(TEST_FILE)

    base_output_dir = Path(BASE_OUTPUT_DIR)

    train_pdb_dir = base_output_dir / "train" / "pdb"
    train_cif_dir = base_output_dir / "train" / "cif"
    test_pdb_dir = base_output_dir / "test" / "pdb"
    test_cif_dir = base_output_dir / "test" / "cif"

    if not train_file.exists():
        raise FileNotFoundError(f"Training file not found: {train_file}")

    if not test_file.exists():
        raise FileNotFoundError(f"Test file not found: {test_file}")

    train_pdb_dir.mkdir(parents=True, exist_ok=True)
    train_cif_dir.mkdir(parents=True, exist_ok=True)
    test_pdb_dir.mkdir(parents=True, exist_ok=True)
    test_cif_dir.mkdir(parents=True, exist_ok=True)

    safe_print("=== TRAIN SET ===")
    download_set(
        train_file,
        train_pdb_dir,
        train_cif_dir,
        overwrite=OVERWRITE,
        max_workers=MAX_WORKERS,
    )

    safe_print("\n=== TEST SET ===")
    download_set(
        test_file,
        test_pdb_dir,
        test_cif_dir,
        overwrite=OVERWRITE,
        max_workers=MAX_WORKERS,
    )


if __name__ == "__main__":
    main()