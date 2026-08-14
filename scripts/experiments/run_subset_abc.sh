#!/usr/bin/env bash
# Regenerate Subsets A, B, C (overhaul.md §1).
#
# Runs the three builders sequentially against the live app, logging each to
# logs/. Rows are flushed to their CSV as they are accepted, so an interruption
# at any point leaves a valid partial dataset.
#
#   ./scripts/run_subset_abc.sh            # fresh run (truncates existing CSVs)
#   RESUME=1 ./scripts/run_subset_abc.sh   # continue after an interruption
#
# ⚠️ Only pass RESUME=1 to continue an interrupted run of THIS configuration.
# The pre-regeneration CSVs share a schema with the new Subset A and B, so
# resuming onto them would merge the old dataset into the new one. Archive at
# evals/data/archive/pre-reingest/.
set -uo pipefail

cd "$(dirname "$0")/.."
API_URL="${API_URL:-http://localhost:8000}"
PY=.venv/bin/python
mkdir -p logs

RESUME_FLAG=""
if [[ "${RESUME:-0}" == "1" ]]; then
    RESUME_FLAG="--resume"
    echo "[$(date -Is)] RESUME mode: continuing from existing CSVs"
fi

run_subset() {
    local name="$1"; shift
    local log="logs/datagen_subset_${name}.log"
    echo "[$(date -Is)] === Subset ${name^^} starting -> $log"
    if "$@" >>"$log" 2>&1; then
        echo "[$(date -Is)] === Subset ${name^^} finished"
        return 0
    fi
    local rc=$?
    echo "[$(date -Is)] !!! Subset ${name^^} FAILED (exit $rc). Last lines:"
    tail -n 15 "$log"
    # A panel outage aborts on purpose; whatever was accepted is already on
    # disk. Stop rather than starting the next subset into the same outage.
    if grep -q "PanelUnavailableError" "$log"; then
        echo "[$(date -Is)] Panel unavailable — stopping. Rerun with RESUME=1 once the API recovers."
        exit "$rc"
    fi
    return "$rc"
}

run_subset a $PY -m evals._dataset_gen.build_subset_a \
    --api-url "$API_URL" --output evals/data/subset_a.csv --count 150 --seed 42 $RESUME_FLAG

run_subset b $PY -m evals._dataset_gen.build_subset_b \
    --output evals/data/subset_b.csv --count 160 $RESUME_FLAG

run_subset c $PY -m evals._dataset_gen.build_subset_c \
    --output evals/data/subset_c.csv --count 200 $RESUME_FLAG

echo "[$(date -Is)] === All three subsets done."
for s in a b c; do
    if [[ -f "evals/data/subset_${s}.csv" ]]; then
        echo "  subset_${s}.csv: $(( $(wc -l < "evals/data/subset_${s}.csv") - 1 )) rows"
    fi
done
