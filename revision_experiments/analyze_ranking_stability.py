#!/usr/bin/env python3
"""
Ranking-stability analysis for the evaluator-reliability study (no API calls).

Uses existing revision_experiments/traces/evaluator_reliability.json only.
"""
from __future__ import annotations

import argparse
import json
import statistics
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    from scipy.stats import spearmanr

    def spearman(x: List[float], y: List[float]) -> Tuple[float, float | None]:
        rho, pval = spearmanr(x, y)
        return float(rho), float(pval)

except ImportError:

    def rankdata(values: List[float]) -> List[float]:
        n = len(values)
        order = sorted(range(n), key=lambda i: values[i])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and values[order[j + 1]] == values[order[i]]:
                j += 1
            avg_rank = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg_rank
            i = j + 1
        return ranks

    def spearman(x: List[float], y: List[float]) -> Tuple[float, float | None]:
        rx, ry = rankdata(x), rankdata(y)
        mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
        num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
        den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
        return (num / den if den else 0.0), None


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "revision_experiments" / "traces" / "evaluator_reliability.json"
    DEFAULT_MD = ROOT / "revision_experiments" / "traces" / "experiment_evaluator_ranking_stability.md"
    DEFAULT_JSON = ROOT / "revision_experiments" / "traces" / "evaluator_ranking_stability.json"


def load_traces(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    traces = []
    for item in data["per_trace"]:
        t = item["trace"]
        scores = [
            r["score"]
            for r in sorted(item["grader_runs"], key=lambda x: x["repeat"])
        ]
        traces.append({
            "trace_id": t["trace_id"],
            "problem_index": t["problem_index"],
            "stratum": t["stratum"],
            "scores": scores,
        })
    return traces


def ranks_descending(scores: List[float]) -> List[int]:
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    ranks = [0] * len(scores)
    for rank, idx in enumerate(order, start=1):
        ranks[idx] = rank
    return ranks


def analyze(traces: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(traces)
    repeats = len(traces[0]["scores"])
    score_matrix = [[tr["scores"][r] for tr in traces] for r in range(repeats)]

    spearman_pairs = []
    for r1, r2 in combinations(range(repeats), 2):
        rho, pval = spearman(score_matrix[r1], score_matrix[r2])
        spearman_pairs.append({"repeat_a": r1 + 1, "repeat_b": r2 + 1, "rho": rho, "p_value": pval})
    rhos = [p["rho"] for p in spearman_pairs]

    opt_idx = [i for i, t in enumerate(traces) if t["stratum"] == "optimal"]
    sub_idx = [i for i, t in enumerate(traces) if t["stratum"] == "suboptimal"]
    total_pairs = len(opt_idx) * len(sub_idx)

    per_repeat_opt_sub = []
    for r in range(repeats):
        wins = 0.0
        for o in opt_idx:
            for s in sub_idx:
                if score_matrix[r][o] > score_matrix[r][s]:
                    wins += 1
                elif score_matrix[r][o] == score_matrix[r][s]:
                    wins += 0.5
        per_repeat_opt_sub.append({
            "repeat": r + 1,
            "fraction_optimal_above_suboptimal": wins / total_pairs if total_pairs else 0.0,
        })

    all5 = never = mixed = 0
    flipped_pairs = []
    for o in opt_idx:
        for s in sub_idx:
            outcomes = [score_matrix[r][o] > score_matrix[r][s] for r in range(repeats)]
            if all(outcomes):
                all5 += 1
            elif not any(outcomes):
                never += 1
            else:
                mixed += 1
                flipped_pairs.append({
                    "optimal": traces[o]["trace_id"],
                    "suboptimal": traces[s]["trace_id"],
                    "optimal_scores": traces[o]["scores"],
                    "suboptimal_scores": traces[s]["scores"],
                })

    rank_matrix = [ranks_descending(score_matrix[r]) for r in range(repeats)]
    per_trace_ranks = []
    for i, tr in enumerate(traces):
        rr = [rank_matrix[r][i] for r in range(repeats)]
        per_trace_ranks.append({
            "trace_id": tr["trace_id"],
            "stratum": tr["stratum"],
            "ranks": rr,
            "rank_sd": statistics.pstdev(rr) if len(rr) > 1 else 0.0,
            "rank_range": max(rr) - min(rr),
        })

    top1_by_repeat = []
    for r in range(repeats):
        best_i = max(range(n), key=lambda i: score_matrix[r][i])
        top1_by_repeat.append({
            "repeat": r + 1,
            "trace_id": traces[best_i]["trace_id"],
            "score": score_matrix[r][best_i],
        })
    unique_top1 = len({x["trace_id"] for x in top1_by_repeat})

    perfect_boundary_cross = []
    for tr in traces:
        hits_ten = [s >= 10.0 - 1e-9 for s in tr["scores"]]
        if any(hits_ten) and not all(hits_ten):
            perfect_boundary_cross.append({
                "trace_id": tr["trace_id"],
                "stratum": tr["stratum"],
                "scores": tr["scores"],
            })

    return {
        "num_traces": n,
        "repeats": repeats,
        "spearman_pairs": spearman_pairs,
        "spearman_mean": sum(rhos) / len(rhos),
        "spearman_min": min(rhos),
        "spearman_max": max(rhos),
        "opt_sub_pairwise": {
            "num_optimal": len(opt_idx),
            "num_suboptimal": len(sub_idx),
            "total_pairs": total_pairs,
            "per_repeat_fraction_optimal_above_suboptimal": per_repeat_opt_sub,
            "fraction_consistent_all_repeats": all5 / total_pairs if total_pairs else 0.0,
            "fraction_never_optimal_above_suboptimal": never / total_pairs if total_pairs else 0.0,
            "fraction_mixed_across_repeats": mixed / total_pairs if total_pairs else 0.0,
            "mixed_examples": flipped_pairs[:10],
        },
        "per_trace_rank_stability": sorted(
            per_trace_ranks, key=lambda x: x["rank_range"], reverse=True
        ),
        "top1_winner_by_repeat": top1_by_repeat,
        "unique_top1_winners": unique_top1,
        "traces_crossing_perfect_score_boundary": perfect_boundary_cross,
        "rollout_winner_note": (
            "This corpus stores one execution trace per problem, not multiple competing "
            "MCTS rollout candidates graded repeatedly. Rollout-level winner-flip rates "
            "require re-scoring 3+ candidates per problem (e.g., from equal-budget or "
            "MCTS candidate logs)."
        ),
    }


def write_markdown(results: Dict[str, Any], path: Path) -> None:
    lines = [
        "# Evaluator Ranking Stability (Post-hoc, No API)",
        "",
        "Analysis of the 150 existing grader scores (30 traces × 5 repeats). "
        "Ranking stability is closer to what MCTS cares about than absolute score variance.",
        "",
        "## Spearman rank correlation (30-trace rankings between repeat pairs)",
        "",
        f"- Mean ρ across 10 repeat pairs: **{results['spearman_mean']:.3f}**",
        f"- Range: **{results['spearman_min']:.3f}** – **{results['spearman_max']:.3f}**",
        "",
        "| Repeat A | Repeat B | ρ |",
        "|----------|----------|---|",
    ]
    for sp in results["spearman_pairs"]:
        lines.append(f"| {sp['repeat_a']} | {sp['repeat_b']} | {sp['rho']:.3f} |")

    os_ = results["opt_sub_pairwise"]
    lines.extend([
        "",
        "## Optimal vs suboptimal pairwise ranking",
        "",
        f"For each repeat, fraction of ({os_['num_optimal']}×{os_['num_suboptimal']} = "
        f"{os_['total_pairs']}) optimal–suboptimal pairs where the optimal trace scores higher:",
        "",
    ])
    for row in os_["per_repeat_fraction_optimal_above_suboptimal"]:
        lines.append(f"- Repeat {row['repeat']}: **{row['fraction_optimal_above_suboptimal']:.1%}**")

    lines.extend([
        "",
        f"- Optimal beats suboptimal in **all 5** repeats: **{os_['fraction_consistent_all_repeats']:.1%}** of pairs",
        f"- Optimal **never** beats suboptimal: **{os_['fraction_never_optimal_above_suboptimal']:.1%}** of pairs",
        f"- **Mixed** (ranking flips across repeats): **{os_['fraction_mixed_across_repeats']:.1%}** of pairs",
        "",
        "## Per-trace rank movement (most unstable)",
        "",
    ])
    for row in results["per_trace_rank_stability"][:8]:
        lines.append(
            f"- `{row['trace_id']}` ({row['stratum']}): ranks={row['ranks']}, range={row['rank_range']}"
        )

    lines.extend([
        "",
        "## Top-1 trace across corpus",
        "",
        f"- Unique top-scoring trace across 5 repeats: **{results['unique_top1_winners']}**",
    ])
    for row in results["top1_winner_by_repeat"]:
        lines.append(f"- Repeat {row['repeat']}: `{row['trace_id']}` (score={row['score']})")

    lines.extend([
        "",
        "## Perfect-score (10) boundary crossings",
        "",
        f"- Traces crossing 10.0 across repeats: **{len(results['traces_crossing_perfect_score_boundary'])}**",
    ])
    for row in results["traces_crossing_perfect_score_boundary"]:
        lines.append(f"- `{row['trace_id']}` ({row['stratum']}): scores={row['scores']}")

    lines.extend([
        "",
        "## Rollout winner stability",
        "",
        results["rollout_winner_note"],
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Ranking stability from reliability JSON")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()

    traces = load_traces(args.input)
    results = analyze(traces)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(results, args.output_md)
    print(json.dumps({
        "spearman_mean": results["spearman_mean"],
        "opt_sub_consistent_all5": results["opt_sub_pairwise"]["fraction_consistent_all_repeats"],
        "opt_sub_mixed": results["opt_sub_pairwise"]["fraction_mixed_across_repeats"],
        "unique_top1": results["unique_top1_winners"],
        "perfect_boundary_crossings": len(results["traces_crossing_perfect_score_boundary"]),
    }, indent=2))
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_md}")


if __name__ == "__main__":
    main()
