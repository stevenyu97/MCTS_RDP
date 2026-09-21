#!/usr/bin/env python3
"""
Build a fixed execution-trace corpus for the evaluator-reliability study.

Archived result JSONs store aggregate scores only (no stdout/stderr). This script
selects stratified NLP4LP problems, generates one representative one-shot program
per problem, executes it once, and caches the trace for repeated LLM grading.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from client_utils import get_openai_client
from revision_experiments.common import (
    DEFAULT_DATASET,
    DEFAULT_EVAL_JSON,
    REVISION_DIR,
    TOLERANCE,
    actual_stratum,
    deterministic_checks,
    generate_oneshot_code,
    load_eval_summaries,
    load_problems,
    pick_stratified_indices,
)
import llm_utils


def main():
    parser = argparse.ArgumentParser(description="Build stratified trace corpus")
    parser.add_argument("--output", type=Path, default=REVISION_DIR / "trace_corpus.jsonl")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--eval-json", type=Path, default=DEFAULT_EVAL_JSON)
    parser.add_argument("--target-per-stratum", type=int, default=15)
    parser.add_argument("--max-pool", type=int, default=80, help="Max problems to try")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    client = get_openai_client()
    problems = load_problems(args.dataset)
    eval_rows = load_eval_summaries(args.eval_json)

    seed_indices = pick_stratified_indices(
        eval_rows, per_stratum=args.target_per_stratum, seed=args.seed
    )
    extra_indices = list(range(1, min(len(problems), args.max_pool) + 1))
    candidate_indices = []
    seen = set()
    for idx in seed_indices + extra_indices:
        if idx not in seen and 1 <= idx <= len(problems):
            candidate_indices.append(idx)
            seen.add(idx)
        if len(candidate_indices) >= args.max_pool:
            break

    args.output.parent.mkdir(parents=True, exist_ok=True)
    collected: dict[str, list] = defaultdict(list)
    records = []

    for problem_index in candidate_indices:
        counts = {k: len(v) for k, v in collected.items()}
        if all(counts.get(k, 0) >= args.target_per_stratum for k in ("optimal", "suboptimal", "failed")):
            break

        item = problems[problem_index - 1]
        print(f"\n=== Problem {problem_index}: generating one-shot trace ===")
        code, gen_tokens = generate_oneshot_code(client, item["problem_description"])
        trace, executed_ok, returncode = llm_utils.execute_code(code, timeout=30)
        det = deterministic_checks(trace, returncode, item["ground_truth"], TOLERANCE)
        stratum = actual_stratum(det)

        if len(collected[stratum]) >= args.target_per_stratum:
            print(f"  Skipping (already have enough {stratum} traces)")
            continue

        record = {
            "trace_id": f"nlp4lp_{problem_index:03d}",
            "problem_index": problem_index,
            "problem_description": item["problem_description"],
            "ground_truth": item["ground_truth"],
            "code": code,
            "execution_trace": trace,
            "returncode": returncode,
            "generation_tokens": gen_tokens,
            "stratum": stratum,
            "deterministic": det,
        }
        records.append(record)
        collected[stratum].append(record)
        print(f"  stratum={stratum} returncode={returncode} det_score={det['score']:.2f}")

    with open(args.output, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    summary = {
        "total_traces": len(records),
        "by_stratum": {k: len(v) for k, v in collected.items()},
        "output": str(args.output),
    }
    summary_path = args.output.with_suffix(".summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Trace corpus built ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
