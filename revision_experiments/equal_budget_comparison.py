#!/usr/bin/env python3
"""
Equal-budget comparison (Reviewer #6 and extended studies).

Each method gets N independent candidates per problem. Report mean-of-N and best-of-N
under the same LLM evaluator used in the paper.

Released Table 5 traces include only:
  oneshot        — one-shot baseline
  mcts_rdp       — full MCTS search on every problem (3 rollouts)
  mcts_rdp_stpa  — search-then-apply (MCTS-RDP-10)

Other method names remain callable but are not part of the released Table 5 archive.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from client_utils import get_openai_client
from revision_experiments.common import (
    BLOCK_SIZE,
    A_SEARCH_COUNT,
    DEFAULT_DATASET,
    REVISION_DIR,
    TOLERANCE,
    BlockExemplarCache,
    SEARCH_FRACTION,
    aggregate_candidate_runs,
    configure_stpa_protocol,
    generate_oneshot_code,
    grade_code,
    load_problems,
    paper_protocol_condition,
    run_cot_once,
    run_mcts_rollouts_once,
    run_mcts_search_then_apply,
    run_self_refine_once,
    sample_dataset_indices,
)

TABLE5_METHODS = ("oneshot", "mcts_rdp", "mcts_rdp_stpa")
ALL_METHODS = ("oneshot", "cot", "self_refine", "mcts_rdp", "mcts_rdp_stpa")


def run_method(
    client,
    method: str,
    problem_description: str,
    ground_truth: str,
    num_candidates: int,
    *,
    dataset_problem_index: int,
    exemplar_cache: BlockExemplarCache | None = None,
) -> Dict[str, Any]:
    if method == "mcts_rdp":
        rollout_runs, _ = run_mcts_rollouts_once(
            client, problem_description, ground_truth, num_rollouts=num_candidates
        )
        agg = aggregate_candidate_runs(rollout_runs)
        agg["mcts_protocol"] = "full_search"
        return agg

    if method == "mcts_rdp_stpa":
        if exemplar_cache is None:
            raise ValueError("exemplar_cache required for mcts_rdp_stpa")
        agg = run_mcts_search_then_apply(
            client,
            dataset_problem_index,
            problem_description,
            ground_truth,
            num_candidates,
            exemplar_cache,
            TOLERANCE,
        )
        agg["mcts_protocol"] = "search_then_apply"
        return agg

    runs: List[Dict[str, Any]] = []
    for cand in range(num_candidates):
        print(f"    candidate {cand + 1}/{num_candidates}")
        t0 = time.time()
        if method == "oneshot":
            code, gen_tokens = generate_oneshot_code(client, problem_description)
            graded = grade_code(client, code, problem_description, ground_truth, TOLERANCE)
            graded["tokens"] = gen_tokens + graded["tokens"]
        elif method == "cot":
            code, gen_tokens = run_cot_once(client, problem_description)
            graded = grade_code(client, code, problem_description, ground_truth, TOLERANCE)
            graded["tokens"] = gen_tokens + graded["tokens"]
        elif method == "self_refine":
            _, total_tokens, graded = run_self_refine_once(
                client, problem_description, ground_truth, TOLERANCE
            )
            graded["tokens"] = total_tokens
        else:
            raise ValueError(f"Unknown method: {method}")

        graded["candidate"] = cand + 1
        graded["time"] = time.time() - t0
        runs.append(graded)

    return aggregate_candidate_runs(runs)


def select_problems(
    all_problems: List[Dict[str, str]],
    num_problems: int,
    random_sample: bool,
    seed: int,
    problem_indices: List[int] | None,
) -> List[Tuple[int, Dict[str, str]]]:
    """Return list of (dataset_problem_index_1based, problem_dict)."""
    dataset_size = len(all_problems)
    if problem_indices:
        indices = problem_indices
    elif random_sample:
        indices = sample_dataset_indices(num_problems, dataset_size, seed)
    else:
        indices = list(range(1, min(num_problems, dataset_size) + 1))

    return [(i, all_problems[i - 1]) for i in indices]


def write_markdown_table(overall: Dict[str, Any], path: Path, methods: List[str]) -> None:
    n = overall["candidates_per_method"]
    lines = [
        "# Equal-Budget Comparison",
        "",
        f"**Problems:** {overall['num_problems']} "
        f"({'random sample' if overall.get('random_sample') else 'first-N'}"
        f"{', seed=' + str(overall['sample_seed']) if overall.get('random_sample') else ''})",
        f"**Candidates per method:** {n}",
        "**Evaluator:** GPT-5 LLM grader (Appendix A)",
        "",
        "| Method | Mean score | Best score | Mean success | Best success | "
        "Mean optimality | Best optimality | Avg tokens |",
        "|--------|------------|------------|--------------|--------------|"
        "-----------------|-----------------|------------|",
    ]
    for method in methods:
        if method not in overall["methods"]:
            continue
        s = overall["methods"][method]
        label = method
        if method == "mcts_rdp":
            label = "MCTS-RDP (full search)"
        elif method == "mcts_rdp_stpa":
            label = "MCTS-RDP (search-then-apply)"
        lines.append(
            f"| {label} | {s['mean_score']:.2f} | {s['best_score']:.2f} | "
            f"{s['mean_success']:.1%} | {s['best_success']:.1%} | "
            f"{s['mean_optimality']:.1%} | {s['best_optimality']:.1%} | {s['avg_tokens']:.0f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def rebuild_exemplar_cache(detailed: List[Dict[str, Any]]) -> BlockExemplarCache:
    cache = BlockExemplarCache()
    for block in sorted(detailed, key=lambda b: b["dataset_problem_index"]):
        mcts = block.get("methods", {}).get("mcts_rdp_stpa")
        if mcts and mcts.get("mcts_condition") == "SEARCH":
            cache.update_from_search_runs(mcts["block_id"], mcts.get("runs", []))
    return cache


def main():
    parser = argparse.ArgumentParser(description="Equal-budget method comparison")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--num-problems", type=int, default=30)
    parser.add_argument("--candidates", type=int, default=3)
    parser.add_argument("--random-sample", action="store_true")
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument(
        "--problem-indices-file",
        type=Path,
        default=None,
        help="JSON file with list of 1-based dataset indices (overrides random/first-N)",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(TABLE5_METHODS),
        choices=list(ALL_METHODS),
    )
    parser.add_argument("--output-json", type=Path, default=REVISION_DIR / "equal_budget_results.json")
    parser.add_argument(
        "--output-md",
        type=Path,
        default=REVISION_DIR / "experiment_equal_budget.md",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--block-size",
        type=int,
        default=30,
        help="Block size for search-then-apply (default 30)",
    )
    parser.add_argument(
        "--search-fraction",
        type=float,
        default=0.10,
        help="Fraction of each block that uses MCTS search (default 0.10 = 10%%)",
    )
    args = parser.parse_args()

    configure_stpa_protocol(args.block_size, args.search_fraction)
    # Refresh module constants after configure (local imports are stale otherwise).
    import revision_experiments.common as stpa
    block_size = stpa.BLOCK_SIZE
    search_fraction = stpa.SEARCH_FRACTION
    search_per_block = stpa.A_SEARCH_COUNT
    print(
        f"STPA protocol: block_size={block_size}, "
        f"search={search_per_block}/{block_size} ({100*search_per_block/block_size:.1f}%)"
    )

    client = get_openai_client()
    all_problems = load_problems(args.dataset)

    indices_from_file = None
    if args.problem_indices_file and args.problem_indices_file.exists():
        indices_from_file = json.loads(args.problem_indices_file.read_text())

    selected = select_problems(
        all_problems,
        args.num_problems,
        args.random_sample,
        args.sample_seed,
        indices_from_file,
    )

    # Search-then-apply requires dataset order within blocks.
    if "mcts_rdp_stpa" in args.methods:
        selected = sorted(selected, key=lambda x: x[0])

    existing: Dict[str, Any] = {}
    if args.resume and args.output_json.exists():
        with open(args.output_json, "r", encoding="utf-8") as f:
            existing = json.load(f)

    detailed = existing.get("detailed", [])
    exemplar_cache = rebuild_exemplar_cache(detailed) if "mcts_rdp_stpa" in args.methods else None

    done_pairs = set()
    for block in detailed:
        dpi = block["dataset_problem_index"]
        for method in args.methods:
            if method in block.get("methods", {}):
                done_pairs.add((dpi, method))

    for run_idx, (dataset_idx, item) in enumerate(selected, start=1):
        print(f"\n================ Run {run_idx}/{len(selected)} | dataset #{dataset_idx} ================")
        if "mcts_rdp_stpa" in args.methods:
            print(f"  STPA slot: {paper_protocol_condition(dataset_idx)}")

        block = next((b for b in detailed if b["dataset_problem_index"] == dataset_idx), None)
        if block is None:
            block = {
                "run_index": run_idx,
                "dataset_problem_index": dataset_idx,
                "methods": {},
            }
            detailed.append(block)
        block["run_index"] = run_idx

        for method in args.methods:
            if args.resume and (dataset_idx, method) in done_pairs:
                print(f"  {method}: skipped (resume)")
                continue

            print(f"  Method: {method}")
            agg = run_method(
                client,
                method,
                item["problem_description"],
                item["ground_truth"],
                args.candidates,
                dataset_problem_index=dataset_idx,
                exemplar_cache=exemplar_cache,
            )
            block["methods"][method] = agg

            payload_partial = {
                "num_problems": len(selected),
                "dataset_problem_indices": [x[0] for x in selected],
                "random_sample": args.random_sample,
                "sample_seed": args.sample_seed,
                "candidates_per_method": args.candidates,
                "methods_list": args.methods,
                "block_size": block_size,
                "search_fraction": search_fraction,
                "search_per_block": search_per_block,
                "detailed": detailed,
            }
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            with open(args.output_json, "w", encoding="utf-8") as f:
                json.dump(payload_partial, f, indent=2, ensure_ascii=False)

    results_by_method: Dict[str, List[Dict[str, Any]]] = {m: [] for m in args.methods}
    for block in sorted(detailed, key=lambda b: b["dataset_problem_index"]):
        for method in args.methods:
            if method in block.get("methods", {}):
                results_by_method[method].append(block["methods"][method])

    overall_methods = {}
    for method in args.methods:
        rows = results_by_method[method]
        if not rows:
            continue
        overall_methods[method] = {
            "mean_score": float(np.mean([r["mean_score"] for r in rows])),
            "best_score": float(np.mean([r["best_score"] for r in rows])),
            "mean_success": float(np.mean([r["mean_executed"] for r in rows])),
            "best_success": float(np.mean([r["best_executed"] for r in rows])),
            "mean_optimality": float(np.mean([r["mean_matches_ground_truth"] for r in rows])),
            "best_optimality": float(np.mean([r["best_matches_ground_truth"] for r in rows])),
            "mean_constraint": float(np.mean([r["mean_constraint_satisfied"] for r in rows])),
            "best_constraint": float(np.mean([r["best_constraint_satisfied"] for r in rows])),
            "avg_tokens": float(np.mean([r["total_tokens"] for r in rows])),
        }

    payload = {
        "num_problems": len(selected),
        "dataset_problem_indices": [x[0] for x in selected],
        "random_sample": args.random_sample,
        "sample_seed": args.sample_seed,
        "candidates_per_method": args.candidates,
        "block_size": block_size,
        "search_fraction": search_fraction,
        "search_per_block": search_per_block,
        "search_percent": round(100.0 * search_per_block / block_size, 2),
        "methods_list": args.methods,
        "methods": overall_methods,
        "detailed": sorted(detailed, key=lambda b: b["dataset_problem_index"]),
    }

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    write_markdown_table(payload, args.output_md, args.methods)
    print("\n=== Equal-budget comparison complete ===")
    print(json.dumps(overall_methods, indent=2))
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_md}")


if __name__ == "__main__":
    main()
