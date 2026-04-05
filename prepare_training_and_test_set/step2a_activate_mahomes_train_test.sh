# save results of predictions within /media/Data/pinmymetal sets/....
#!/usr/bin/env bash
set -euo pipefail
### SETTINGS ###

job_root="/media/Data/pinmymetal_sets/mahomes/train_set"

pdb_dir="/media/Data/pinmymetal_sets/train/pdb_updatedv2"

N_JOBS=4
MAHOMES_DIR="/home/mechti/MAHOMES-II"
VENV="$MAHOMES_DIR/venv/bin/activate"
pdbids_query_txt="$job_root/pdbids_query.txt"
################
# Validate inputs


if [[ ! -d "$pdb_dir" ]]; then
    echo "[ERROR] PDB directory not found: $pdb_dir"
    exit 1
fi

echo "[INFO] PDB dir:  $pdb_dir"

mkdir -p "$job_root"

# 0) Build IDs list automatically from existing PDB files
find "$pdb_dir" -maxdepth 1 -type f -name "*.pdb" -printf '%f\n' \
    | sed 's/\.pdb$//' \
    | sort -u > "$pdbids_query_txt"

if [[ ! -s "$pdbids_query_txt" ]]; then
    echo "[ERROR] No .pdb files found in: $pdb_dir"
    exit 1
fi

echo "[INFO] Auto-generated IDs file: $pdbids_query_txt ($(wc -l < "$pdbids_query_txt") IDs)"

# 1) Clean old batch_input_part_* so no stale IDs are used
rm -f "$job_root"/batch_input_part_* 2>/dev/null || true

# 2) Split current IDs file into N_JOBS parts
split -d -n l/$N_JOBS "$pdbids_query_txt" "$job_root/batch_input_part_"

job_index=0
declare -a pids

for part_file in "$job_root"/batch_input_part_*; do
    job_dir="$job_root/job_$job_index"

    # *** IMPORTANT CHANGE: do NOT delete job_dir anymore ***
    # rm -rf "$job_dir"           # ← removed
    mkdir -p "$job_dir"           # just ensure it exists, keep old content

    (
        exec >> "$job_dir/job.log" 2>&1

        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting job $job_index (PID $$)"
        echo "[INFO] Processing files from: $part_file"

        copied=0
        missing=0
        skipped_copy=0

        # IFS + trimming CR/spaces/tabs on each ID line
        while IFS= read -r struct_id_raw; do
            # Trim trailing \r, spaces, and tabs
            struct_id="${struct_id_raw%%[$'\r\t ']*}"

            [[ -z "$struct_id" ]] && continue

            pdb_file="$pdb_dir/${struct_id}.pdb"
            target_pdb="$job_dir/${struct_id}.pdb"

            if [[ -f "$target_pdb" ]]; then
                # PDB for this structure already in job_dir → no need to recopy
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

        # batch_input.txt reflects the IDs list that driver.sh will see
        cp "$part_file" "$job_dir/batch_input.txt"

        # 4) Run MAHOMES driver
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
echo "[DONE] All parallel MAHOMES jobs finished at $(date '+%Y-%m-%d %H:%M:%S')"
