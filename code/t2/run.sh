#!/bin/bash

set -euo pipefail

EXEC=${1:?"Usage: $0 <executable> [--arg=val ...]"}
shift

read -ra NPROCS <<< "${MAL_NPROCS:-1 2 4 8 16 32 48 64}"
ITERS=${MAL_ITERS:-10}
TASKS_PER_NODE=${MAL_TASKS_PER_NODE:-0}
BENCH_CSV=${MAL_BENCH_CSV:-0}
export MAL_BENCH_CSV="$BENCH_CSV"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CSV="${SCRIPT_DIR}/${EXEC}_data.csv"

mkdir -p "${SCRIPT_DIR}/out"

if [[ ! -f "$CSV" ]]; then

	if [[ "$BENCH_CSV" -eq 1 ]]; then

		echo "example,variant,mode,np,work_items,compute_seconds,errors,iter,exit_code" > "$CSV"

	else

		echo "exec,nproc,iter,seconds,exit_code" > "$CSV"

	fi

fi

module load cesga/2025 gcc/14.3.0 openmpi/5.0.9
make clean
make BENCH_CSV="$BENCH_CSV"

chmod +x "${SCRIPT_DIR}/build/${EXEC}"

for nproc in "${NPROCS[@]}"; do

	if [[ "$nproc" -le 0 ]]; then

		echo "[WARN] Skipping invalid nproc=${nproc}" >&2
		continue

	fi

	SBATCH_ARGS=(
		-n "$nproc"
		-J "TFM.${EXEC}.${nproc}"
		-o "${SCRIPT_DIR}/out/${EXEC}.${nproc}.o"
		-e "${SCRIPT_DIR}/out/${EXEC}.${nproc}.e"
	)

	if [[ "$TASKS_PER_NODE" -gt 0 ]]; then

		SBATCH_ARGS+=(--ntasks-per-node "$TASKS_PER_NODE")

	fi

	JOB_ID=$(sbatch --parsable "${SBATCH_ARGS[@]}" "${SCRIPT_DIR}/job.sh" "$CSV" "$nproc" "$ITERS" "${SCRIPT_DIR}/build/${EXEC}" "$@")
	echo "[SUBMIT] job_id=${JOB_ID} exec=${EXEC} nproc=${nproc} iters=${ITERS}"

done
