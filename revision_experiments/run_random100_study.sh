#!/usr/bin/env bash
# Table 5: 100 random NLP4LP problems, 3 candidates each.
# Methods: one-shot, MCTS-RDP (full search), MCTS-RDP-10 (search-then-apply).
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "ERROR: export OPENAI_API_KEY first." >&2
  exit 1
fi

TRACE_DIR="revision_experiments/traces"
INDICES_FILE="$TRACE_DIR/random100_problem_indices.json"
OUTPUT_JSON="$TRACE_DIR/equal_budget_random100.json"

python3 - <<'PY'
import json
from pathlib import Path
from revision_experiments.common import sample_dataset_indices, load_problems, DEFAULT_DATASET

out = Path("revision_experiments/traces/random100_problem_indices.json")
if not out.exists():
    n = len(load_problems(DEFAULT_DATASET))
    indices = sample_dataset_indices(100, n, seed=42)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(indices, indent=2))
    print(f"Wrote {out} ({len(indices)} indices)")
else:
    print(f"Using existing {out}")
PY

python3 revision_experiments/equal_budget_comparison.py \
  --num-problems 100 \
  --random-sample \
  --sample-seed 42 \
  --problem-indices-file "$INDICES_FILE" \
  --candidates 3 \
  --methods oneshot mcts_rdp mcts_rdp_stpa \
  --block-size 30 \
  --search-fraction 0.10 \
  --output-json "$OUTPUT_JSON" \
  --resume

echo "Done: $OUTPUT_JSON"
