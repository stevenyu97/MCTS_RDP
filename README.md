# MCTS-RDP

Code and archived results for *Pre-hoc Scaling via Monte Carlo Search over Reasoning Decomposition Plans for Optimization Code Synthesis*.

The released repository contains both **complete** benchmark runs and **partial** archived snapshots. Complete artifacts are used for table reconstruction where available; partial files are retained for transparency but should not be interpreted as full-benchmark results. In particular, the released GPT-5 and Gemini NLP4LP MCTS logs contain all 242 instances, whereas some baseline artifacts cover only subsets of the corresponding benchmark.

Each released artifact below is labeled **complete**, **complete archived run**, **partial**, or **auxiliary**. Complete artifacts reconstruct the corresponding printed table row via `aggregate_results.py`. Complete archived runs contain every benchmark instance but do **not** reconstruct every printed column.

```bash
pip install -r requirements.txt
python3 aggregate_results.py --agg best
```

Revision-round scripts (Tables 4–5) live in `revision_experiments/`. Traces they produce go in `revision_experiments/traces/`.

`aggregate_results.py` averages the archived problem-level fields `success_rate`, `avg_score`, `constraint_satisfaction_rate`, `ground_truth_match_rate`, `avg_tokens`, and `avg_time` / `avg_time_sec`. Optimality attainment is computed from the stored LLM `ground_truth_match_rate` field.

## File-to-experiment mapping

**Datasets.** For MCTS files, `_aq` denotes `all_questions.jsonl` (NLP4LP, 242 instances). Note that the root-level `baseline_aq.json` is instead a 100-problem one-shot archive; the separate `gemini/baseline_aq.json` contains 242 NLP4LP instances.

### Complete

These files reconstruct the printed row (rounding only).

| Table | Method | File | n |
|-------|--------|------|---|
| Table 2 | MCTS-RDP GPT-5 (search-then-apply) | `evaluationMCTS_results.json` | 242 |
| Table 2 | MCTS-RDP Gemini (search-then-apply) | `gemini/evaluationMCTS_results_gem.json` | 242 |
| Table 2 | One-shot Gemini | `gemini/baseline_aq.json` | 242 |
| Table 7 | Decomposition-only (w/o MCTS) | `ablation_no_mcts_results.json` | 242 |

### Complete archived runs (not exact printed-row reconstruction)

These files contain all 242 NLP4LP instances. They are retained as complete archives, but `aggregate_results.py` does **not** recover every printed Table 2 column.

| Table | Method | File | n | Notes |
|-------|--------|------|---|-------|
| Table 2 | Optimus Gemini | `gemini/optimus_baseline_gem_results.json` | 242 | Success/reward are close; constraint, tokens, and time differ from the printed row (optimality 69.83% vs printed 69.42%) |
| Table 2 | CoT GPT-5 | `cot_results.json` | 242 | Optimality matches 54.13%; success, reward, constraint, tokens, and time differ |

### Partial

| Table / role | Method | File | n | Notes |
|--------------|--------|------|---|-------|
| Table 1 | One-shot GPT-5 | `baseline_aq.json` | 100 | Not NLP4LP |
| Table 1 | Optimus GPT-5 | `optimus_baseline_results.json` | 100 | |
| Table 2 one-shot GPT-5 | One-shot GPT-5 | `evaluation_results.json` | 100 | Printed Table 2 row is n=242 |
| — | MCTS full search GPT-5 | `evaluationMCTS_results_aq.json` | 242 | Not Table 1 NL4OPT |
| — | MCTS full search Gemini | `gemini/evaluationMCTS_results_aq_gem.json` | 1 | Stub |

### Auxiliary (`revision_experiments/traces/`)

| Experiment | File | Contents |
|------------|------|----------|
| Table 5 equal best-of-N (seed 42, 100 NLP4LP) | `revision_experiments/traces/equal_budget_random100.json` | Traces for **one-shot**, **MCTS-RDP**, and **MCTS-RDP-10** only |
| Table 5 problem indices | `revision_experiments/traces/random100_problem_indices.json` | Seed-42 sample |
| Table 4 evaluator reliability | `revision_experiments/traces/evaluator_reliability.json`, `evaluator_ranking_stability.json`, `trace_corpus.jsonl` | 30 fixed traces × 5 GPT-5 grades |
| Aggregation output | `revision_experiments/traces/aggregated_summary.json` | From `aggregate_results.py` |
| 30-problem equal-budget pilot | `revision_experiments/traces/equal_budget_results.json` | One-shot and full MCTS-RDP only; not Table 5 |

Table 5 does **not** include CoT, Self-Refine, or OPTIMUS traces.

### Missing from this release

Table 1 NL4OPT MCTS JSON; Self-Refine logs; GPT-5 one-shot/Optimus n=242; Gemini CoT; wireless result JSON; Table 6 \(k\)-sweep JSON.
