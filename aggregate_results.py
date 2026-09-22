#!/usr/bin/env python3
"""
Unified result aggregation for MCTS-RDP experiments.

Rebuilds manuscript-style summary tables from per-problem JSON logs.
Supports best-of-N vs mean aggregation and SEARCH / APPLY splits.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

BLOCK_SIZE = 24
SEARCH_PER_BLOCK = 3


def load_results(path: Path) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "results" in data:
        return data["results"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unsupported JSON structure in {path}")


def _get(row: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    for k in keys:
        if k in row:
            return row[k]
    return default


def per_problem_metrics(
    row: Dict[str, Any],
    agg: str = "best",
) -> Dict[str, float]:
    """Derive per-problem metrics under best or mean aggregation."""
    all_results = row.get("all_results")
    if all_results and isinstance(all_results, list) and len(all_results) > 0:
        scores = [float(x) for x in all_results]
        if agg == "mean":
            reward = sum(scores) / len(scores)
        else:
            reward = max(scores)
        # success: any run with score >= 0 (executed)
        success = 1.0 if any(s >= 0 for s in scores) else 0.0
    else:
        reward = float(_get(row, "avg_score", default=-1.0))
        success = float(_get(row, "success_rate", default=0.0))

    return {
        "reward": reward,
        "success": success,
        "constraint": float(_get(row, "constraint_satisfaction_rate", default=0.0)),
        "optimality": float(_get(row, "ground_truth_match_rate", default=0.0)),
        "tokens": float(_get(row, "avg_tokens", default=0.0)),
        "time": float(_get(row, "avg_time", "avg_time_sec", default=0.0)),
    }


def summarize_rows(
    rows: List[Dict[str, Any]],
    agg: str = "best",
    condition: Optional[str] = None,
) -> Dict[str, Any]:
    if condition is not None:
        rows = [r for r in rows if _get(r, "condition", default="").upper() == condition.upper()]

    if not rows:
        return {"n": 0}

    metrics = [per_problem_metrics(r, agg=agg) for r in rows]
    rewards = [m["reward"] for m in metrics]
    mean_reward = sum(rewards) / len(rewards)
    var = sum((r - mean_reward) ** 2 for r in rewards) / max(1, len(rewards) - 1) if len(rewards) > 1 else 0.0

    return {
        "n": len(rows),
        "success_rate_pct": 100.0 * sum(m["success"] for m in metrics) / len(metrics),
        "reward_mean": mean_reward,
        "reward_sd": math.sqrt(var),
        "constraint_pct": 100.0 * sum(m["constraint"] for m in metrics) / len(metrics),
        "optimality_pct": 100.0 * sum(m["optimality"] for m in metrics) / len(metrics),
        "tokens_mean": sum(m["tokens"] for m in metrics) / len(metrics),
        "time_mean": sum(m["time"] for m in metrics) / len(metrics),
    }


def block_protocol_stats(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Report search-then-apply block statistics."""
    n = len(rows)
    n_blocks = math.ceil(n / BLOCK_SIZE) if n else 0
    search_indices = set()
    for b in range(n_blocks):
        for i in range(SEARCH_PER_BLOCK):
            idx = b * BLOCK_SIZE + i + 1  # 1-based problem_index
            if idx <= n:
                search_indices.add(idx)

    search_rows = [r for r in rows if int(_get(r, "problem_index", default=0)) in search_indices]
    apply_rows = [r for r in rows if int(_get(r, "problem_index", default=0)) not in search_indices]

    # Also use condition field when present
    cond_search = [r for r in rows if _get(r, "condition", default="").upper() == "SEARCH"]
    cond_apply = [r for r in rows if _get(r, "condition", default="").upper() in ("EXAMPLES_ONLY", "APPLY")]

    return {
        "block_size": BLOCK_SIZE,
        "search_per_block": SEARCH_PER_BLOCK,
        "total_problems": n,
        "num_blocks": n_blocks,
        "search_by_index": len(search_rows),
        "apply_by_index": len(apply_rows),
        "search_fraction_by_index": len(search_rows) / n if n else 0.0,
        "search_by_condition_field": len(cond_search),
        "apply_by_condition_field": len(cond_apply),
    }


def format_summary(label: str, s: Dict[str, Any]) -> str:
    if s.get("n", 0) == 0:
        return f"{label}: (no rows)"
    return (
        f"{label} (n={s['n']}): "
        f"success={s['success_rate_pct']:.2f}%, "
        f"reward={s['reward_mean']:.2f}±{s['reward_sd']:.2f}, "
        f"constraint={s['constraint_pct']:.2f}%, "
        f"optimality={s['optimality_pct']:.2f}%, "
        f"tokens={s['tokens_mean']:.0f}, "
        f"time={s['time_mean']:.2f}s"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate MCTS-RDP experiment JSON logs.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--agg", choices=["best", "mean"], default="best")
    parser.add_argument("--output", type=Path, default=None, help="Write JSON summary here")
    args = parser.parse_args()

    root = args.root
    datasets = {
        "NLP4LP_MCTS_STPA": root / "evaluationMCTS_results.json",
        "NLP4LP_MCTS_full": root / "evaluationMCTS_results_aq.json",  # MCTS_OPS.py on all_questions; not NL4OPT
        "NLP4LP_MCTS_Gemini_STPA": root / "gemini" / "evaluationMCTS_results_gem.json",
        "NLP4LP_MCTS_Gemini_full": root / "gemini" / "evaluationMCTS_results_aq_gem.json",
        "NLP4LP_oneshot_first100": root / "evaluation_results.json",  # n=100; likely first 100 of all_questions
        "problems100_oneshot": root / "baseline_aq.json",  # baseline.py → problems_100.jsonl
        "CoT_NLP4LP": root / "cot_results.json",
        "Ablation_no_MCTS": root / "ablation_no_mcts_results.json",
        "problems100_Optimus": root / "optimus_baseline_results.json",
        "NLP4LP_oneshot_Gemini": root / "gemini" / "baseline_aq.json",
        "NLP4LP_Optimus_Gemini": root / "gemini" / "optimus_baseline_gem_results.json",
    }

    report: Dict[str, Any] = {"aggregation": args.agg, "datasets": {}}

    print(f"=== MCTS-RDP Result Aggregation (agg={args.agg}) ===\n")

    for name, path in datasets.items():
        if not path.exists():
            print(f"[skip] {name}: {path} not found")
            continue
        rows = load_results(path)
        overall = summarize_rows(rows, agg=args.agg)
        search = summarize_rows(rows, agg=args.agg, condition="SEARCH")
        apply_rows = [
            r for r in rows
            if _get(r, "condition", default="").upper() in ("EXAMPLES_ONLY", "APPLY")
        ]
        apply_ = summarize_rows(apply_rows, agg=args.agg) if apply_rows else {"n": 0}

        try:
            rel_path = str(path.relative_to(root))
        except ValueError:
            rel_path = str(path)
        entry = {
            "path": rel_path,
            "protocol": block_protocol_stats(rows),
            "overall": overall,
            "search_only": search,
            "apply_only": apply_,
        }
        report["datasets"][name] = entry

        print(f"--- {name} ({path.name}) ---")
        print(format_summary("Overall", overall))
        if search.get("n", 0):
            print(format_summary("SEARCH", search))
        if apply_.get("n", 0):
            print(format_summary("APPLY", apply_))
        proto = entry["protocol"]
        print(
            f"Block protocol: B={proto['block_size']}, k={proto['search_per_block']}, "
            f"search={proto['search_by_condition_field']}/{proto['total_problems']} "
            f"({100*proto['search_fraction_by_index']:.1f}% by index)"
        )
        print()

    # Also show mean aggregation for NLP4LP if all_results available
    nlp_path = root / "evaluationMCTS_results.json"
    if nlp_path.exists():
        rows = load_results(nlp_path)
        mean_overall = summarize_rows(rows, agg="mean")
        print("--- NLP4LP best vs mean (from stored per-problem summaries) ---")
        print(format_summary("Best (stored)", summarize_rows(rows, agg="best")))
        print(format_summary("Mean (if all_results present)", mean_overall))
        print()

    out = args.output or (root / "revision_experiments" / "traces" / "aggregated_summary.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote summary to {out}")


if __name__ == "__main__":
    main()
