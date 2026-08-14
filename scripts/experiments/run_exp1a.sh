#!/usr/bin/env bash
# Run Experiment 1a — safety classification (overhaul.md §3).
#
# Scores every guard in the roster on Subset B and the two external held-out
# injection sets, then the zero-shot LLM baseline on Subset B alone.
#
#   ./scripts/run_exp1a.sh                        # everything available
#   SYSTEMS=prompt_guard ./scripts/run_exp1a.sh   # one guard
#   NO_BASELINE=1 ./scripts/run_exp1a.sh          # guards only, no API spend
#
# Preconditions are checked and reported before anything is spent. A guard that
# cannot be reached is skipped by the runner itself, so a missing fine-tuned
# checkpoint or an unscoped HF token costs a warning rather than the run.
set -uo pipefail

cd "$(dirname "$0")/.."
PY=.venv/bin/python

# The HF token and OpenRouter key live in .env, which the app loads but a bare
# shell does not.
if [[ -f .env ]]; then
    set -a; source .env; set +a
fi
mkdir -p logs evals/data/results

DATASET="${DATASET:-evals/data/subset_b.csv}"
HELDOUT="${HELDOUT:-evals/data/heldout_injection_en.csv,evals/data/heldout_injection_id.csv}"
SYSTEMS="${SYSTEMS:-all}"
GUARD_URL="${GUARD_URL:-http://localhost:7998}"
GUARD_FT_URL="${GUARD_FT_URL:-http://localhost:7999}"
REPEATS="${REPEATS:-3}"
# Qwen3Guard is served locally through Ollama (docker compose --profile qwen).
# 0.6B only by default — each row is one generation call per prompt across every
# dataset; append larger variants (e.g. a converted 4B) once they are pulled.
QWEN_MODELS="${QWEN_MODELS:-qwen3guard-gen-0.6b}"
QWEN_BASE_URL="${QWEN_BASE_URL:-http://localhost:11434/v1}"
QWEN_API_KEY="${QWEN_API_KEY:-ollama}"
# A provider suffix is only meaningful on the HF router; empty for local Ollama.
case "$QWEN_BASE_URL" in
    *huggingface*) QWEN_PROVIDER="${QWEN_PROVIDER:-featherless-ai}" ;;
    *) QWEN_PROVIDER="${QWEN_PROVIDER:-}" ;;
esac
OUTPUT_CSV="${OUTPUT_CSV:-evals/data/results/exp1a_safety.csv}"
LOG="logs/exp1a_safety.log"

echo "[$(date -Is)] === Experiment 1a preflight"

fail=0
check_file() {
    if [[ -f "$1" ]]; then
        # Counted through a CSV reader, not wc: these files contain quoted
        # newlines (document excerpts, multi-line injections), so a line count
        # overstates the row count by a factor of three.
        local rows
        rows=$("$PY" -c "import csv,sys; print(sum(1 for _ in csv.DictReader(open(sys.argv[1], newline='', encoding='utf-8'))))" "$1")
        echo "  ok      $1 ($rows rows)"
    else
        echo "  MISSING $1"
        [[ "${2:-required}" == "required" ]] && fail=1
    fi
}

check_service() {
    local name="$1" url="$2" severity="${3:-required}"
    local code
    code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 "$url/health" 2>/dev/null)
    if [[ "$code" == "200" ]]; then
        echo "  ok      $name at $url"
        # A guard serving a different transformers version than the one the
        # model was trained and evaluated on is a different classifier. Measured:
        # 4.57.1 vs 5.9.0 flips 8 of 160 rows on Subset B, moving off-the-shelf
        # detection from 0.4250 to 0.3500 with no other change. The image pins
        # the version; this checks the *running* container honours the pin.
        local want served
        want=$(grep -oP '(?<=^transformers==)\S+' services/prompt_guard/requirements.txt 2>/dev/null)
        served=$(curl -s --max-time 5 "$url/health" | grep -oP '(?<="transformers":")[^"]+' 2>/dev/null)
        if [[ -n "$want" && -n "$served" && "$want" != "$served" ]]; then
            echo "  STALE   $name runs transformers $served, pinned $want — rebuild it"
            echo "          (docker compose build $name && docker compose up -d $name)"
            fail=1
        fi
    else
        echo "  DOWN    $name at $url (HTTP ${code:-no response})"
        [[ "$severity" == "required" ]] && fail=1
    fi
}

check_file "$DATASET"
IFS=',' read -ra heldout_paths <<<"$HELDOUT"
for path in "${heldout_paths[@]}"; do
    [[ -n "$path" ]] && check_file "$path" optional
done

# The off-the-shelf guard is the row the experiment cannot do without. The
# fine-tuned one is optional on purpose: it does not exist until a training run
# has produced it, and the rest of the roster is still worth measuring.
case ",$SYSTEMS," in
    *,all,*|*,prompt_guard,*) check_service "prompt-guard" "$GUARD_URL" ;;
esac
case ",$SYSTEMS," in
    *,all,*|*,prompt_guard_ft,*) check_service "prompt-guard-ft" "$GUARD_FT_URL" optional ;;
esac

if [[ "${NO_BASELINE:-0}" != "1" ]]; then
    if [[ -n "${OPENROUTER_API_KEY:-}" ]]; then
        echo "  ok      OPENROUTER_API_KEY set (baseline, ${REPEATS} passes)"
    else
        echo "  MISSING OPENROUTER_API_KEY — the baseline row will be skipped"
    fi
fi
case ",$SYSTEMS," in
    *,all,*|*,qwen*,*)
        # A guard that is simply down would otherwise post a perfect score
        # (both clients fail closed), so the runner probes before scoring — but
        # flag an unreachable endpoint here too, before any spend.
        code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 "${QWEN_BASE_URL%/v1}/api/tags" 2>/dev/null)
        if [[ "$code" == "200" ]]; then
            echo "  ok      Qwen endpoint at $QWEN_BASE_URL ($QWEN_MODELS)"
        else
            echo "  DOWN    Qwen endpoint at $QWEN_BASE_URL — the Qwen3Guard rows will be skipped"
            echo "          (docker compose --profile qwen up -d ollama)"
        fi
        ;;
esac

if [[ "$fail" != "0" ]]; then
    echo "[$(date -Is)] !!! preflight failed — nothing was run"
    exit 1
fi

baseline_flag=""
[[ "${NO_BASELINE:-0}" == "1" ]] && baseline_flag="--no-baseline"

echo "[$(date -Is)] === Running -> $LOG"
"$PY" -m evals.exp1a_safety.run \
    --dataset "$DATASET" \
    --heldout "$HELDOUT" \
    --systems "$SYSTEMS" \
    --guard-url "$GUARD_URL" \
    --guard-ft-url "$GUARD_FT_URL" \
    --baseline-repeats "$REPEATS" \
    --qwen-models "$QWEN_MODELS" \
    --qwen-base-url "$QWEN_BASE_URL" \
    --qwen-api-key "$QWEN_API_KEY" \
    --qwen-provider "$QWEN_PROVIDER" \
    --output-csv "$OUTPUT_CSV" \
    $baseline_flag 2>&1 | tee "$LOG"

rc=${PIPESTATUS[0]}
if [[ "$rc" != "0" ]]; then
    echo "[$(date -Is)] !!! Experiment 1a FAILED (exit $rc) — see $LOG"
    exit "$rc"
fi

echo "[$(date -Is)] === Experiment 1a finished"
echo
sed -n '/^  SUMMARY$/,$p' "$LOG"
