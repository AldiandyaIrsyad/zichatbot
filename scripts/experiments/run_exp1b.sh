#!/usr/bin/env bash
# Run Experiment 1b — IVM relevance / out-of-domain detection (overhaul.md §3).
#
# Compares a keyword-overlap baseline against the production relevance backends
# (LLM-judge, similarity-threshold, NLI-entailment) and the offline centroid
# detector (Metode 4), on Subset C.
#
#   ./scripts/run_exp1b.sh                       # everything available
#   SWEEP=1 ./scripts/run_exp1b.sh               # sweep similarity/NLI/centroid thresholds
#   SKIP_JUDGE=1 ./scripts/run_exp1b.sh          # no LLM-judge (no API spend)
#
# The three threshold backends have uncalibrated placeholder defaults, so their
# single-point numbers are not comparable to a tuned judge. SWEEP=1 runs each
# over a small grid so an operating point can be chosen from the curve rather
# than trusted from the placeholder.
#
# Preconditions are checked and reported before anything is spent; a backend
# whose dependency is down is skipped rather than aborting the run.
set -uo pipefail

cd "$(dirname "$0")/.."
PY=.venv/bin/python

if [[ -f .env ]]; then
    set -a; source .env; set +a
fi
mkdir -p logs evals/data/results

DATASET="${DATASET:-evals/data/subset_c.csv}"
API_URL="${API_URL:-http://localhost:8000}"
INFINITY_URL="${INFINITY_URL:-http://localhost:7997}"
JUDGE_REPEATS="${JUDGE_REPEATS:-3}"
OUTPUT_CSV="${OUTPUT_CSV:-evals/data/results/exp1b_relevance.csv}"
LOG="logs/exp1b_relevance.log"

SIM_GRID="${SIM_GRID:-0.0 0.01 0.02 0.05 0.1}"
NLI_GRID="${NLI_GRID:-0.1 0.3 0.5 0.7 0.9}"
CENTROID_GRID="${CENTROID_GRID:-0.2 0.3 0.4 0.5 0.6}"

echo "[$(date -Is)] === Experiment 1b preflight"

fail=0
if [[ -f "$DATASET" ]]; then
    rows=$("$PY" -c "import csv,sys; print(sum(1 for _ in csv.DictReader(open(sys.argv[1], newline='', encoding='utf-8'))))" "$DATASET")
    echo "  ok      $DATASET ($rows rows)"
else
    echo "  MISSING $DATASET"; fail=1
fi

check_http() {
    local name="$1" url="$2" severity="${3:-required}"
    local code
    code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 "$url" 2>/dev/null)
    if [[ "$code" =~ ^(200|404)$ ]]; then
        echo "  ok      $name reachable"
    else
        echo "  DOWN    $name ($url, HTTP ${code:-no response})"
        [[ "$severity" == "required" ]] && fail=1
    fi
}

# KB search drives every retrieval-dependent backend; the app must be up.
check_http "app / KB search" "$API_URL/api/admin/pdfs"
check_http "Infinity (NLI)" "$INFINITY_URL/health" optional
check_http "Qdrant (centroid)" "http://${QDRANT_HOST:-127.0.0.1}:${QDRANT_PORT:-6333}/collections" optional

skip_judge_flag=""
if [[ "${SKIP_JUDGE:-0}" == "1" ]]; then
    skip_judge_flag="--skip-llm-judge"
    echo "  note    LLM-judge skipped (SKIP_JUDGE=1)"
elif [[ -n "${OPENROUTER_API_KEY:-}" ]]; then
    echo "  ok      OPENROUTER_API_KEY set (LLM-judge, ${JUDGE_REPEATS} passes)"
else
    echo "  MISSING OPENROUTER_API_KEY — the LLM-judge row will be skipped"
fi

if [[ "$fail" != "0" ]]; then
    echo "[$(date -Is)] !!! preflight failed — nothing was run"
    exit 1
fi

run_once() {
    local sim="$1" nli="$2" cen="$3"; shift 3
    "$PY" -m evals.exp1b_relevance.run \
        --dataset "$DATASET" \
        --api-url "$API_URL" \
        --infinity-url "$INFINITY_URL" \
        --judge-repeats "$JUDGE_REPEATS" \
        --similarity-threshold "$sim" \
        --nli-threshold "$nli" \
        --centroid-threshold "$cen" \
        --output-csv "$OUTPUT_CSV" \
        $skip_judge_flag "$@"
}

echo "[$(date -Is)] === Running -> $LOG"
{
    if [[ "${SWEEP:-0}" == "1" ]]; then
        # Judge + baseline once (deterministic wrt threshold), then sweep the
        # three threshold backends. Re-running the judge per threshold would
        # burn API budget for no new information.
        echo "### baseline + judge (thresholds do not affect them) ###"
        run_once 0.02 0.5 0.5 --skip-similarity --skip-nli --skip-centroid
        for t in $SIM_GRID; do
            echo "### similarity threshold = $t ###"
            run_once "$t" 0.5 0.5 --skip-llm-judge --skip-nli --skip-centroid
        done
        for t in $NLI_GRID; do
            echo "### nli threshold = $t ###"
            run_once 0.02 "$t" 0.5 --skip-llm-judge --skip-similarity --skip-centroid
        done
        for t in $CENTROID_GRID; do
            echo "### centroid threshold = $t ###"
            run_once 0.02 0.5 "$t" --skip-llm-judge --skip-similarity --skip-nli
        done
    else
        run_once 0.02 0.5 0.5
    fi
} 2>&1 | tee "$LOG"

rc=${PIPESTATUS[0]}
if [[ "$rc" != "0" ]]; then
    echo "[$(date -Is)] !!! Experiment 1b FAILED (exit $rc) — see $LOG"
    exit "$rc"
fi
echo "[$(date -Is)] === Experiment 1b finished"
