# MCTS-RDP

Code and archived results for *Pre-hoc Scaling via Monte Carlo Search over Reasoning Decomposition Plans for Optimization Code Synthesis*.

The released repository contains both **complete** benchmark runs and **partial** archived snapshots. Complete artifacts are used for table reconstruction where available; partial files are retained for transparency but should not be interpreted as full-benchmark results. In particular, the released GPT-5 and Gemini NLP4LP MCTS logs contain all 242 instances, whereas some baseline artifacts cover only subsets of the corresponding benchmark.

Each released artifact below is labeled **complete**, **partial**, or **auxiliary**.

```bash
pip install -r requirements.txt
python3 aggregate_results.py --agg best
```

Revision-round scripts (Tables 4–5) live in `revision_experiments/`. Traces they produce go in `revision_experiments/traces/`.

`aggregate_results.py` averages the archived problem-level fields `success_rate`, `avg_score`, `constraint_satisfaction_rate`, `ground_truth_match_rate`, `avg_tokens`, and `avg_time` / `avg_time_sec`. Optimality is the stored LLM `ground_truth_match_rate` flag.

## File-to-experiment mapping

**Datasets.** `_aq` on MCTS files means `all_questions.jsonl` (NLP4LP, 242), not NL4OPT. `baseline_aq.json` is a 100-problem one-shot.

### Complete

| Table | Method | File | n |
|-------|--------|------|---|
| Table 2 | MCTS-RDP GPT-5 (search-then-apply) | `evaluationMCTS_results.json` | 242 |
| Table 2 | MCTS-RDP Gemini (search-then-apply) | `gemini/evaluationMCTS_results_gem.json` | 242 |
| Table 2 | One-shot Gemini | `gemini/baseline_aq.json` | 242 |
| Table 2 | Optimus Gemini | `gemini/optimus_baseline_gem_results.json` | 242 |
| Table 2 | CoT GPT-5 | `cot_results.json` | 242 |
| Table 7 | Decomposition-only (w/o MCTS) | `ablation_no_mcts_results.json` | 242 |

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
