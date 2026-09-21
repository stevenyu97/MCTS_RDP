#!/usr/bin/env python3
"""
Evaluator-reliability study (Reviewer #3).

Re-grade fixed execution traces multiple times with the paper LLM evaluator,
then compare against deterministic post-hoc checks.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import llm_utils
from client_utils import get_openai_client
from revision_experiments.common import REVISION_DIR, TOLERANCE


def majority_bool(values: List[bool]) -> bool:
    return Counter(values).most_common(1)[0][0]


def agreement_rate(values: List[bool]) -> float:
    if not values:
        return 0.0
    mode_count = Counter(values).most_common(1)[0][1]
    return mode_count / len(values)


def summarize_trace_runs(runs: List[Dict[str, Any]], det: Dict[str, Any]) -> Dict[str, Any]:
    scores = [r["score"] for r in runs]
    mean_score = statistics.mean(scores)
    sd_score = statistics.pstdev(scores) if len(scores) > 1 else 0.0
    score_range = max(scores) - min(scores)

    llm_executed = majority_bool([r["executed"] for r in runs])
    llm_constraint = majority_bool([r["constraint_satisfied"] for r in runs])
    llm_gt = majority_bool([r["matches_ground_truth"] for r in runs])

    return {
        "n_repeats": len(runs),
        "scores": scores,
        "mean_score": mean_score,
        "sd_score": sd_score,
        "score_range": score_range,
        "executed_agreement": agreement_rate([r["executed"] for r in runs]),
        "constraint_agreement": agreement_rate([r["constraint_satisfied"] for r in runs]),
        "gt_agreement": agreement_rate([r["matches_ground_truth"] for r in runs]),
        "llm_majority": {
            "executed": llm_executed,
            "constraint_satisfied": llm_constraint,
            "matches_ground_truth": llm_gt,
        },
        "deterministic": {
            "executed": bool(det["executed"]),
            "constraint_satisfied": bool(det["constraint_satisfied"]),
            "matches_ground_truth": bool(det["matches_ground_truth"]),
            "returncode": det.get("returncode"),
            "timed_out": det.get("timed_out"),
            "score": det.get("score"),
        },
        "llm_vs_det_agreement": {
            "executed": llm_executed == bool(det["executed"]),
            "constraint_satisfied": llm_constraint == bool(det["constraint_satisfied"]),
            "matches_ground_truth": llm_gt == bool(det["matches_ground_truth"]),
        },
    }


def aggregate_all(per_trace: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(per_trace)
    if n == 0:
        return {}

    def mean_field(key: str) -> float:
        return sum(t[key] for t in per_trace) / n

    def agree_field(key: str) -> float:
        return sum(1 for t in per_trace if t["llm_vs_det_agreement"][key]) / n

    return {
        "num_traces": n,
        "repeats_per_trace": per_trace[0]["n_repeats"] if per_trace else 0,
        "mean_score_sd_across_traces": mean_field("sd_score"),
        "mean_score_range_across_traces": mean_field("score_range"),
        "mean_absolute_score_variation": mean_field("score_range"),  # max-min per trace
        "mean_executed_agreement": mean_field("executed_agreement"),
        "mean_constraint_agreement": mean_field("constraint_agreement"),
        "mean_gt_agreement": mean_field("gt_agreement"),
        "llm_vs_det_executed_agreement": agree_field("executed"),
        "llm_vs_det_constraint_agreement": agree_field("constraint_satisfied"),
        "llm_vs_det_gt_agreement": agree_field("matches_ground_truth"),
        "by_stratum": _by_stratum(per_trace),
    }


def _by_stratum(per_trace: List[Dict[str, Any]]) -> Dict[str, Any]:
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for row in per_trace:
        buckets.setdefault(row["stratum"], []).append(row)

    out = {}
    for stratum, rows in buckets.items():
        n = len(rows)
        out[stratum] = {
            "count": n,
            "mean_sd_score": sum(r["sd_score"] for r in rows) / n,
            "mean_score_range": sum(r["score_range"] for r in rows) / n,
            "llm_vs_det_gt_agreement": sum(
                1 for r in rows if r["llm_vs_det_agreement"]["matches_ground_truth"]
            )
            / n,
        }
    return out


def write_markdown_report(summary: Dict[str, Any], per_trace: List[Dict[str, Any]], path: Path):
    lines = [
        "# Evaluator Reliability Study",
        "",
        "## Aggregate metrics",
        "",
        f"- Traces graded: **{summary.get('num_traces', 0)}**",
        f"- Repeats per trace: **{summary.get('repeats_per_trace', 0)}**",
        f"- Mean score SD (across traces): **{summary.get('mean_score_sd_across_traces', 0):.3f}**",
        f"- Mean score range (max−min per trace): **{summary.get('mean_score_range_across_traces', 0):.3f}**",
        f"- Mean boolean agreement (executed): **{summary.get('mean_executed_agreement', 0):.1%}**",
        f"- Mean boolean agreement (constraint_satisfied): **{summary.get('mean_constraint_agreement', 0):.1%}**",
        f"- Mean boolean agreement (matches_ground_truth): **{summary.get('mean_gt_agreement', 0):.1%}**",
        "",
        "## LLM majority vs deterministic post-hoc checks",
        "",
        f"- Executed agreement: **{summary.get('llm_vs_det_executed_agreement', 0):.1%}**",
        f"- Constraint satisfaction agreement: **{summary.get('llm_vs_det_constraint_agreement', 0):.1%}**",
        f"- Ground-truth match agreement: **{summary.get('llm_vs_det_gt_agreement', 0):.1%}**",
        "",
        "## By stratum",
        "",
    ]
    for stratum, stats in summary.get("by_stratum", {}).items():
        lines.append(
            f"- **{stratum}** (n={stats['count']}): mean SD={stats['mean_sd_score']:.3f}, "
            f"mean range={stats['mean_score_range']:.3f}, GT agree={stats['llm_vs_det_gt_agreement']:.1%}"
        )

    lines.extend(["", "## Per-trace detail", ""])
    for row in per_trace:
        lines.append(
            f"- `{row['trace_id']}` ({row['stratum']}): scores={row['scores']}, "
            f"SD={row['sd_score']:.2f}, det GT={row['deterministic']['matches_ground_truth']}, "
            f"LLM GT majority={row['llm_majority']['matches_ground_truth']}"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="LLM evaluator reliability study")
    parser.add_argument(
        "--corpus",
        type=Path,
        default=REVISION_DIR / "trace_corpus.jsonl",
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output-json", type=Path, default=REVISION_DIR / "evaluator_reliability.json")
    parser.add_argument(
        "--output-md",
        type=Path,
        default=REVISION_DIR / "experiment_evaluator_reliability.md",
    )
    parser.add_argument("--model", type=str, default="gpt-5")
    args = parser.parse_args()

    if not args.corpus.exists():
        raise SystemExit(
            f"Trace corpus not found: {args.corpus}\n"
            "Run: python revision_experiments/build_trace_corpus.py"
        )

    client = get_openai_client()
    traces = []
    with open(args.corpus, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                traces.append(json.loads(line))

    all_results = []
    per_trace_summary = []

    for rec in traces:
        print(f"\n=== {rec['trace_id']} ({rec['stratum']}) — {args.repeats} grader repeats ===")
        runs = []
        for rep in range(args.repeats):
            score, explanation, executed, tokens, constraint_ok, matches_gt = (
                llm_utils.evaluate_trace_with_llm(
                    client,
                    rec["problem_description"],
                    rec["execution_trace"],
                    rec["ground_truth"],
                    TOLERANCE,
                    model=args.model,
                )
            )
            runs.append({
                "repeat": rep + 1,
                "score": float(score),
                "executed": bool(executed),
                "constraint_satisfied": bool(constraint_ok),
                "matches_ground_truth": bool(matches_gt),
                "explanation": explanation,
                "tokens": tokens,
            })
            print(f"  repeat {rep + 1}: score={score}, exec={executed}, gt={matches_gt}")

        det = rec["deterministic"]
        summary = summarize_trace_runs(runs, det)
        summary.update({
            "trace_id": rec["trace_id"],
            "problem_index": rec["problem_index"],
            "stratum": rec["stratum"],
        })
        per_trace_summary.append(summary)
        all_results.append({"trace": rec, "grader_runs": runs, "summary": summary})

    aggregate = aggregate_all(per_trace_summary)
    payload = {"aggregate": aggregate, "per_trace": all_results}

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    write_markdown_report(aggregate, per_trace_summary, args.output_md)
    print("\n=== Reliability study complete ===")
    print(json.dumps(aggregate, indent=2))
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_md}")


if __name__ == "__main__":
    main()
