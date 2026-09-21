#!/usr/bin/env bash
# Revision experiments: Table 4 (evaluator reliability) and 30-problem equal-budget pilot.
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "ERROR: export OPENAI_API_KEY before running." >&2
  exit 1
fi

TRACE_DIR="revision_experiments/traces"

echo "=== Step 1: Build trace corpus (one execution per stratified problem) ==="
python3 revision_experiments/build_trace_corpus.py \
  --target-per-stratum 15 \
  --output "$TRACE_DIR/trace_corpus.jsonl"

echo ""
echo "=== Step 2: Evaluator reliability (5 LLM grades per fixed trace) ==="
python3 revision_experiments/evaluator_reliability_study.py \
  --corpus "$TRACE_DIR/trace_corpus.jsonl" \
  --repeats 5 \
  --output-json "$TRACE_DIR/evaluator_reliability.json"

echo ""
echo "=== Step 3: Equal-budget pilot (30 problems × 3 candidates) ==="
python3 revision_experiments/equal_budget_comparison.py \
  --num-problems 30 \
  --candidates 3 \
  --methods oneshot mcts_rdp \
  --output-json "$TRACE_DIR/equal_budget_results.json" \
  --resume

echo ""
echo "Done. Traces are in $TRACE_DIR/"
