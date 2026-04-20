#!/usr/bin/env bash
set -euo pipefail

# Run the external MAHOMES workflow on the held-out raw test PDB set.
#
# This is parallel to `step2a_activate_mahomes_train_test.sh`, but writes to a
# separate `mahomes/test_set` tree so train/test MAHOMES outputs stay isolated.
# The defaults target the paths that currently exist on this machine.

job_root="${JOB_ROOT:-/media/Data/pinmymetal_sets/mahomes/test_set}"
pdb_dir="${PDB_DIR:-/media/Data/pinmymetal_sets/test/pdb}"

N_JOBS="${N_JOBS:-4}"
MAHOMES_DIR="${MAHOMES_DIR:-/home/mechti/MAHOMES-II}"
VENV="${VENV:-$MAHOMES_DIR/venv/bin/activate}"
pdbids_query_txt="$job_root/pdbids_query.txt"

if [[ ! -d "$pdb_dir" ]]; then
    echo "[ERROR] PDB directory not found: $pdb_dir"
    exit 1
fi

if [[ ! -d "$MAHOMES_DIR" ]]; then
    echo "[ERROR] MAHOMES directory not found: $MAHOMES_DIR"
    exit 1
fi

if [[ ! -f "$VENV" ]]; then
    echo "[ERROR] MAHOMES virtualenv activate script not found: $VENV"
    exit 1
fi

echo "[INFO] PDB dir:  $pdb_dir"
echo "[INFO] Job root: $job_root"
echo "[INFO] N_JOBS:   $N_JOBS"

mkdir -p "$job_root"

find "$pdb_dir" -maxdepth 1 -type f -name "*.pdb" -printf '%f\n' \
    | sed 's/\.pdb$//' \
    | sort -u > "$pdbids_query_txt"

if [[ ! -s "$pdbids_query_txt" ]]; then
    echo "[ERROR] No .pdb files found in: $pdb_dir"
    exit 1
fi

echo "[INFO] Auto-generated IDs file: $pdbids_query_txt ($(wc -l < "$pdbids_query_txt") IDs)"

rm -f "$job_root"/batch_input_part_* 2>/dev/null || true
split -d -n "l/$N_JOBS" "$pdbids_query_txt" "$job_root/batch_input_part_"

job_index=0
declare -a pids

for part_file in "$job_root"/batch_input_part_*; do
    job_dir="$job_root/job_$job_index"
    mkdir -p "$job_dir"

    (
        exec >> "$job_dir/job.log" 2>&1

        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting job $job_index (PID $$)"
        echo "[INFO] Processing files from: $part_file"

        copied=0
        missing=0
        skipped_copy=0

        while IFS= read -r struct_id_raw; do
            struct_id="$(printf '%s' "$struct_id_raw" | sed 's/[[:space:]]*$//')"
            [[ -z "$struct_id" ]] && continue

            pdb_file="$pdb_dir/${struct_id}.pdb"
            target_pdb="$job_dir/${struct_id}.pdb"

            if [[ -f "$target_pdb" ]]; then
                echo "[SKIP COPY] PDB already present for ID: '$struct_id'"
                skipped_copy=$((skipped_copy + 1))
            elif [[ -f "$pdb_file" ]]; then
                cp "$pdb_file" "$job_dir/"
                copied=$((copied + 1))
            else
                echo "  [WARN] PDB not found for ID: '$struct_id' (raw: '$struct_id_raw') in $pdb_dir"
                missing=$((missing + 1))
            fi
        done < "$part_file"

        echo "[INFO] Copied $copied PDBs for this job; $skipped_copy copies skipped; $missing IDs missing PDBs."

        cp "$part_file" "$job_dir/batch_input.txt"

        source "$VENV"
        bash "$MAHOMES_DIR/driver.sh" "$job_dir"

        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished job $job_index"
    ) &

    pids[$job_index]=$!
    echo "[LAUNCHED] Job $job_index (PID ${pids[$job_index]}) -> Log: $job_dir/job.log"

    job_index=$((job_index + 1))
done

echo ""
echo "============================================"
echo "All $job_index jobs launched!"
echo "PIDs: ${pids[*]}"
echo "============================================"
echo ""
echo "Monitor with: watch -n 2 'tail -n 3 $job_root/job_*/job.log'"

wait

echo ""
echo "[DONE] All parallel MAHOMES test jobs finished at $(date '+%Y-%m-%d %H:%M:%S')"
