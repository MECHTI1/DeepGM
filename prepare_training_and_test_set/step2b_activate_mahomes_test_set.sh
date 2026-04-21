#!/usr/bin/env bash
# Save MAHOMES predictions for the held-out test-set structures.
set -euo pipefail

### SETTINGS ###
job_root="${JOB_ROOT:-/media/Data/pinmymetal_sets/mahomes/test_set}"
pdb_dir="${PDB_DIR:-/media/Data/pinmymetal_sets/test/pdb_updatedv2}"
N_JOBS="${N_JOBS:-4}"
MAHOMES_DIR="${MAHOMES_DIR:-/home/mechti/MAHOMES-II}"
VENV="${VENV:-$MAHOMES_DIR/venv/bin/activate}"
CLEAN_JOB_DIRS="${CLEAN_JOB_DIRS:-1}"
RUN_MODE="${RUN_MODE:-all}"
pdb_source_marker="$job_root/pdb_source_dir.txt"
pdbids_query_txt="$job_root/pdbids_query.txt"
################

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

case "$RUN_MODE" in
    all|last|features|predict) ;;
    *)
        echo "[ERROR] Invalid RUN_MODE: $RUN_MODE"
        echo "Allowed values: all | last | features | predict"
        exit 1
        ;;
esac

if [[ "$CLEAN_JOB_DIRS" != "0" && "$CLEAN_JOB_DIRS" != "1" ]]; then
    echo "[ERROR] CLEAN_JOB_DIRS must be 0 or 1, got: $CLEAN_JOB_DIRS"
    exit 1
fi

mkdir -p "$job_root"
printf '%s\n' "$pdb_dir" > "$pdb_source_marker"

echo "[INFO] PDB dir:        $pdb_dir"
echo "[INFO] Job root:       $job_root"
echo "[INFO] N_JOBS:         $N_JOBS"
echo "[INFO] RUN_MODE:       $RUN_MODE"
echo "[INFO] CLEAN_JOB_DIRS: $CLEAN_JOB_DIRS"
echo "[INFO] PDB source tag: $pdb_source_marker"

if [[ "$CLEAN_JOB_DIRS" == "1" ]]; then
    echo "[INFO] Existing job_* directories will be recreated so MAHOMES reruns on pdb_updatedv2."
else
    echo "[WARN] Reusing existing job_* directories can keep stale MAHOMES artifacts."
fi

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

    if [[ "$CLEAN_JOB_DIRS" == "1" ]]; then
        rm -rf "$job_dir"
    fi
    mkdir -p "$job_dir"

    (
        exec >> "$job_dir/job.log" 2>&1

        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting job $job_index (PID $$)"
        echo "[INFO] Processing files from: $part_file"

        copied=0
        missing=0
        replaced=0

        while IFS= read -r struct_id_raw; do
            struct_id="$(printf '%s' "$struct_id_raw" | sed 's/[[:space:]]*$//')"
            [[ -z "$struct_id" ]] && continue

            pdb_file="$pdb_dir/${struct_id}.pdb"
            target_pdb="$job_dir/${struct_id}.pdb"

            if [[ -f "$pdb_file" ]]; then
                if [[ -f "$target_pdb" ]]; then
                    replaced=$((replaced + 1))
                fi
                cp -f "$pdb_file" "$target_pdb"
                copied=$((copied + 1))
            else
                echo "  [WARN] PDB not found for ID: '$struct_id' (raw: '$struct_id_raw') in $pdb_dir"
                missing=$((missing + 1))
            fi
        done < "$part_file"

        echo "[INFO] Copied $copied PDBs for this job; $replaced existing job copies overwritten; $missing IDs missing PDBs."

        cp "$part_file" "$job_dir/batch_input.txt"

        source "$VENV"
        bash "$MAHOMES_DIR/driver.sh" "$job_dir" "$RUN_MODE"

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
