#!/usr/bin/env bash
set -euo pipefail

### SETTINGS ###
SOURCE_JOB_ROOT="${SOURCE_JOB_ROOT:-/media/Data/pinmymetal_sets/mahomes/train_set}"
JOB_ROOT="${JOB_ROOT:-/media/Data/pinmymetal_sets/mahomes/train_set_selective}"
PDB_DIR="${PDB_DIR:-/media/Data/pinmymetal_sets/train/pdb_updatedv2}"
N_JOBS="${N_JOBS:-4}"
MAHOMES_DIR="${MAHOMES_DIR:-/home/mechti/MAHOMES-II}"
VENV="${VENV:-$MAHOMES_DIR/venv/bin/activate}"
PYTHON_BIN="${PYTHON_BIN:-/home/mechti/miniconda3/envs/deepgm-py312/bin/python}"
FAILED_IDS_FILE="$JOB_ROOT/pdbids_query_failed_only.txt"
################

if [[ ! -d "$SOURCE_JOB_ROOT" ]]; then
    echo "[ERROR] Source MAHOMES train-set root not found: $SOURCE_JOB_ROOT"
    exit 1
fi

if [[ ! -d "$PDB_DIR" ]]; then
    echo "[ERROR] PDB directory not found: $PDB_DIR"
    exit 1
fi

if [[ ! -f "$VENV" ]]; then
    echo "[ERROR] MAHOMES virtualenv activate script not found: $VENV"
    exit 1
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "[ERROR] Python interpreter not found or not executable: $PYTHON_BIN"
    exit 1
fi

mkdir -p "$JOB_ROOT"

echo "[INFO] Source job root:  $SOURCE_JOB_ROOT"
echo "[INFO] Selective job root: $JOB_ROOT"
echo "[INFO] PDB dir:          $PDB_DIR"

"$PYTHON_BIN" - <<'PY' "$SOURCE_JOB_ROOT" "$FAILED_IDS_FILE"
from __future__ import annotations

from pathlib import Path
import re
import sys

source_root = Path(sys.argv[1])
failed_ids_path = Path(sys.argv[2])

runparty_pattern = re.compile(r"\[ERROR\] runParty3Calc\.sh failed for (.+?) [–-] skipping")
summary_suffixes = (
    "Missing bluues",
    "No metal sites were found in this file.",
    "Invalid structure file input. Please make sure file is valid input for Rosetta 3.13.",
    "no residues within 6 angstroms of site center",
)

failed_ids: set[str] = set()

for log_path in sorted(source_root.glob("job_*/job.log")):
    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = runparty_pattern.search(line)
            if match:
                failed_ids.add(match.group(1).strip())
                continue

            if ".pdb," not in line:
                continue
            if not any(suffix in line for suffix in summary_suffixes):
                continue

            structure_id = line.split(",", 1)[0].strip().strip('"')
            if structure_id.endswith(".pdb"):
                structure_id = structure_id[:-4]
            if structure_id:
                failed_ids.add(structure_id)

failed_ids_path.write_text(
    "".join(f"{structure_id}\n" for structure_id in sorted(failed_ids)),
    encoding="utf-8",
)

print(f"[INFO] Wrote {len(failed_ids)} failed structure IDs to {failed_ids_path}")
PY

if [[ ! -s "$FAILED_IDS_FILE" ]]; then
    echo "[ERROR] No failed structure IDs were collected from: $SOURCE_JOB_ROOT"
    exit 1
fi

echo "[INFO] Failed IDs file: $FAILED_IDS_FILE ($(wc -l < "$FAILED_IDS_FILE") IDs)"

rm -f "$JOB_ROOT"/batch_input_part_* 2>/dev/null || true
split -d -n "l/$N_JOBS" "$FAILED_IDS_FILE" "$JOB_ROOT/batch_input_part_"

job_index=0
declare -a pids

for part_file in "$JOB_ROOT"/batch_input_part_*; do
    job_dir="$JOB_ROOT/job_$job_index"
    mkdir -p "$job_dir"

    (
        exec >> "$job_dir/job.log" 2>&1

        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting selective job $job_index (PID $$)"
        echo "[INFO] Processing failed IDs from: $part_file"

        copied=0
        missing=0
        skipped_copy=0

        while IFS= read -r struct_id_raw; do
            struct_id="$(printf '%s' "$struct_id_raw" | sed 's/[[:space:]]*$//')"
            [[ -z "$struct_id" ]] && continue

            pdb_file="$PDB_DIR/${struct_id}.pdb"
            target_pdb="$job_dir/${struct_id}.pdb"

            if [[ -f "$target_pdb" ]]; then
                echo "[SKIP COPY] PDB already present for ID: '$struct_id'"
                skipped_copy=$((skipped_copy + 1))
            elif [[ -f "$pdb_file" ]]; then
                cp "$pdb_file" "$job_dir/"
                copied=$((copied + 1))
            else
                echo "[WARN] PDB not found for ID: '$struct_id' in $PDB_DIR"
                missing=$((missing + 1))
            fi
        done < "$part_file"

        echo "[INFO] Copied $copied PDBs; $skipped_copy copies skipped; $missing IDs missing PDBs."

        cp "$part_file" "$job_dir/batch_input.txt"

        source "$VENV"
        bash "$MAHOMES_DIR/driver.sh" "$job_dir"

        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished selective job $job_index"
    ) &

    pids[$job_index]=$!
    echo "[LAUNCHED] Selective job $job_index (PID ${pids[$job_index]}) -> Log: $job_dir/job.log"

    job_index=$((job_index + 1))
done

echo ""
echo "============================================"
echo "All $job_index selective jobs launched"
echo "PIDs: ${pids[*]}"
echo "============================================"
echo ""

wait

echo "[DONE] All selective MAHOMES jobs finished at $(date '+%Y-%m-%d %H:%M:%S')"
