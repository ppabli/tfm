#!/bin/bash

#SBATCH --mem 10G
#SBATCH -t 00:15:00
#SBATCH --cpus-per-task=1

set -euo pipefail

CSV=$1
NPROC=$2
ITERS=$3
EXEC=$4
shift 4

ITER_TIMEOUT_SEC=${MAL_ITER_TIMEOUT_SEC:-0}
CONTINUE_ON_ERROR=${MAL_CONTINUE_ON_ERROR:-0}
BENCH_CSV_MODE=${MAL_BENCH_CSV:-0}

EXEC_NAME=$(basename "$EXEC")
LOCK="${CSV}.lock"
OUT_FILE=$(mktemp "${TMPDIR:-/tmp}/malleable_job_output.XXXXXX")

trap 'rm -f "$OUT_FILE"' EXIT

export OMPI_MCA_pml=ob1
export OMPI_MCA_btl=self,vader,tcp

for i in $(seq 1 "$ITERS"); do

	START_NS=$(date +%s%N)
	: > "$OUT_FILE"

	if [[ "$ITER_TIMEOUT_SEC" -gt 0 ]]; then

		if timeout --foreground "${ITER_TIMEOUT_SEC}s" mpirun -np "$NPROC" --mca orte_base_help_aggregate 0 "$EXEC" "$@" 2>&1 | tee "$OUT_FILE"; then

			EXIT_CODE=0

		else

			EXIT_CODE=$?

		fi

	else

		if mpirun -np "$NPROC" --mca orte_base_help_aggregate 0 "$EXEC" "$@" 2>&1 | tee "$OUT_FILE"; then

			EXIT_CODE=0

		else

			EXIT_CODE=$?

		fi

	fi

	END_NS=$(date +%s%N)
	ELAPSED=$(echo "scale=6; ($END_NS - $START_NS) / 1000000000" | bc)
	BENCH_LINE=$(awk '/^CSV,/{line=$0} END{if (line != "") print line}' "$OUT_FILE")

	if [[ -n "$BENCH_LINE" ]]; then

		RESULT_LINE="${BENCH_LINE#CSV,},${i},${EXIT_CODE}"

	else

		if [[ "$BENCH_CSV_MODE" -eq 1 ]]; then

			ERRORS_FIELD=0

			if [[ "$EXIT_CODE" -ne 0 ]]; then

				ERRORS_FIELD=1

			fi

			RESULT_LINE="${EXEC_NAME},unknown,unknown,${NPROC},0,${ELAPSED},${ERRORS_FIELD},${i},${EXIT_CODE}"

		else

			RESULT_LINE="${EXEC_NAME},${NPROC},${i},${ELAPSED},${EXIT_CODE}"

		fi

	fi

	(
		flock 200
		printf '%s\n' "$RESULT_LINE" >> "$CSV"
	) 200>"$LOCK"

	if [[ "$EXIT_CODE" -ne 0 ]]; then

		echo "[JOB] mpirun failed exec=${EXEC_NAME} nproc=${NPROC} iter=${i} exit_code=${EXIT_CODE}" >&2

		if [[ "$CONTINUE_ON_ERROR" -ne 1 ]]; then

			exit "$EXIT_CODE"

		fi

	fi

done
