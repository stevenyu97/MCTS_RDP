

import json
import os
import time
from typing import Optional
import numpy as np
from pathlib import Path
from openai import OpenAI

import llm_utils
from mcts import SharedNodeMCTS
from typing import List, Dict, Optional, Any


# =========================
# Config
# =========================
DATASET_PATH = "/home/ubuntu/LLM_Planning/all_questions.jsonl"
#DATASET_PATH = "/home/ubuntu/LLM_Planning/problems_100.jsonl"

RESULTS_JSON = "evaluationMCTS_results.json"
BLOCK_SIZE = 24            # 24 problems per block
A_SEARCH_COUNT =  3       # first 3 => full search, last 21 => examples-only
SELF_REFINE_MAX_RETRIES = 2
REWARD_THRESHOLD = 1
EX_CACHE_DIR = Path("examples_cache")
EX_CACHE_DIR.mkdir(exist_ok=True)

# Prefer env var; avoid hardcoding secrets
client = OpenAI(api_key="")

# =========================
# Exemplar helpers (best-in-block)
# =========================
_block_best = {}  # {block_id: {"reward": float, "tokens": int, "example": str}}

def _compact_prompt_path(prompt_score_path, keep=3):
    """
    Convert a (prompt, score) path into a compact exemplar string.
    keep: how many steps to retain for token efficiency.
    """
    trimmed = prompt_score_path[:keep]
    lines = []
    for i, (p, s) in enumerate(trimmed, 1):
        lines.append(f"[Step {i}] score={s:.2f}\n{p.strip()}")
    return "\n".join(lines)

def _should_replace_block_best(existing: Optional[Dict[str, Any]], new_reward: float, new_tokens: int) -> bool:
    """
    Replacement policy for block-best exemplar:
    1) Higher reward is better
    2) If rewards tie (within exact equality), fewer tokens is better
    """
    if existing is None:
        return True
    if new_reward > existing["reward"]:
        return True
    if new_reward < existing["reward"]:
        return False
    # tie on reward -> prefer lower tokens
    return new_tokens < existing["tokens"]

def save_block_best_example(block_id: int, reward: float, tokens: int, path):
    exemplar = _compact_prompt_path(path, keep=3)
    prev = _block_best.get(block_id)
    if _should_replace_block_best(prev, reward, tokens):
        _block_best[block_id] = {"reward": float(reward), "tokens": int(tokens), "example": exemplar}
        with open(EX_CACHE_DIR / f"ex_block_{block_id:04d}.json", "w", encoding="utf-8") as f:
            json.dump(_block_best[block_id], f, ensure_ascii=False, indent=2)

def load_block_best_example(block_id: int) -> List[str]:
    """
    Returns a single-element list with the best exemplar text for the block, or [] if none.
    """
    if block_id in _block_best:
        return [_block_best[block_id]["example"]]
    p = EX_CACHE_DIR / f"ex_block_{block_id:04d}.json"
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            obj = json.load(f)
            return [obj["example"]]
    return []

def build_examples_only_prompt(problem_description: str, exemplar: List[str]) -> str:
    """
    Build the one-shot prompt for the EXAMPLES-ONLY condition using at most one exemplar.
    """
    head = (
        "You are a Python optimization expert.\n"
        "Use the compact exemplar below to structure the solution. Keep code concise and runnable.\n\n"
        "=== EXEMPLAR (best-in-block) ===\n"
    )
    ex_block = exemplar[0] if exemplar else "[No exemplar available]\n"
    tail = (
        "\n=== TARGET PROBLEM ===\n"
        f"{problem_description.strip()}\n\n"
        "Produce a complete, runnable solution. Print key results and FINAL_ANSWER if applicable."
    )
    return head + ex_block + tail

# =========================
# Data loading
# =========================
problems = []
with open(DATASET_PATH, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        problems.append({
            "problem_description": obj["en_question"],
            "ground_truth": str(obj["en_answer"])
        })

# Ensure results file exists (we’ll store a flat list of dicts)
if not os.path.exists(RESULTS_JSON):
    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump([], f)

# =========================
# Experiment loop
# =========================
all_results = []
all_tokens = []

for idx, item in enumerate(problems, start=1):
    print(f"\n================ Problem {idx}/{len(problems)} ================")
    problem_description = item["problem_description"]
    ground_truth = item["ground_truth"]
    tolerance = 1e-4

    # Block routing
    block_id = (idx - 1) // BLOCK_SIZE
    within_blk = (idx - 1) % BLOCK_SIZE
    use_search = within_blk < A_SEARCH_COUNT  # first 5 => search; last 5 => examples-only

    results = []
    success_count = 0
    token_counts = []
    time_counts = []
    constraint_satisfied_count = 0
    ground_truth_match_count = 0
    per_problem_candidates = []  # store candidates for selection

    if use_search:
        # ======== FULL SEARCH branch (3 runs) ========
        mcts = SharedNodeMCTS()

        for run in range(3):
            try:
                print(f"\n=== Iteration {run + 1} (SEARCH) ===")
                run_start_time = time.time()
                total_tokens = 0

                sentences, token_used = llm_utils.generate_structured_sentences(client, problem_description)
                total_tokens += token_used

                cumulative_code = ""
                prompt_score_path = []
                current_node = mcts.root
                parent_at_depth = {0: mcts.root}

                for depth, sentence in enumerate(sentences):
                    parent_node = parent_at_depth[depth]

                    if parent_node.children:
                        current_node = mcts.select_node_by_uct(parent_node)
                        parent_at_depth[depth + 1] = current_node
                    else:
                        current_node = parent_node

                    is_last = (depth == len(sentences) - 1)
                    temp_prompt, token_used = llm_utils.select_prompt_from_llm(client, sentence, depth=depth, is_last=is_last)
                    total_tokens += token_used
                    temp_score, token_used = llm_utils.score_prompt(client, temp_prompt)
                    total_tokens += token_used
                    temp_score = float(temp_score)

                    found_sibling = None
                    for sibling in parent_node.children:
                        if sibling.score == temp_score:
                            found_sibling = sibling
                            break

                    if found_sibling:
                        current_node = found_sibling
                        prompt = current_node.prompt
                        score = current_node.score
                        print(f"Reusing node at depth {depth} | Score: {score:.2f} | Prompt: {prompt!r}")
                    else:
                        current_node = mcts.expand_node(parent_node, temp_prompt, temp_score)
                        prompt = temp_prompt
                        score = temp_score
                        print(f"First expansion at depth {depth} | Score: {score:.2f} | Prompt: {prompt!r}")

                    parent_at_depth[depth + 1] = current_node

                    code, token_used = llm_utils.generate_code_from_prompt(client, prompt, previous_code=cumulative_code)
                    total_tokens += token_used
                    cumulative_code = code
                    prompt_score_path.append((prompt, score))

                # Evaluate
                full_code = "# === Auto-generated Optimization Script ===\n\n" + cumulative_code
                reward, explanation, success, token_used, constraint_satisfied, matches_ground_truth = llm_utils.get_final_reward_from_output(
                    full_code, problem_description, client, ground_truth=ground_truth, tolerance=tolerance
                )
                total_tokens += token_used
                reward_score = float(reward)
                print(explanation, reward_score)

                # Self-refine
                attempt = 0
                while (not success or reward_score < REWARD_THRESHOLD) and attempt < SELF_REFINE_MAX_RETRIES:
                    print(f"\n🔁 Self-Refine Attempt {attempt + 1}")
                    revised_code, token_used = llm_utils.revise_code_based_on_feedback(client, full_code, explanation, problem_description)
                    total_tokens += token_used
                    reward, explanation, success, token_used, constraint_satisfied, matches_ground_truth = llm_utils.get_final_reward_from_output(
                        revised_code, problem_description, client, ground_truth=ground_truth, tolerance=tolerance
                    )
                    total_tokens += token_used
                    reward_score = float(reward)
                    print(f"Tokens used: {total_tokens}, Reward: {reward_score}")
                    full_code = revised_code
                    attempt += 1

                if success:
                    success_count += 1
                    results.append(reward_score)
                else:
                    results.append(-1.0)
                if constraint_satisfied:
                    constraint_satisfied_count += 1
                if matches_ground_truth:
                    ground_truth_match_count += 1

                run_time = time.time() - run_start_time
                time_counts.append(run_time)

                per_problem_candidates.append({
                    "run": run + 1,
                    "reward": reward_score,
                    "success": bool(success),
                    "constraint_satisfied": bool(constraint_satisfied),
                    "matches_ground_truth": bool(matches_ground_truth),
                    "tokens": total_tokens,
                    "time": run_time,
                    "code": full_code,
                    "path": prompt_score_path,  # needed for exemplar
                })
                # MCTS backprop
                leaf_node = mcts.build_path(prompt_score_path)
                mcts.backpropagate_reward(leaf_node, reward_score)
                token_counts.append(total_tokens)

            except Exception as e:
                print(f"Run {run + 1} failed with error: {e}")
                results.append(-1.0)
                run_time = time.time() - run_start_time
                time_counts.append(run_time)

        successful = [c for c in per_problem_candidates if c["success"]]
        best = max(successful, key=lambda c: c["reward"]) if successful else max(per_problem_candidates, key=lambda c: c["reward"])

        # Update the single best exemplar for the block: compare (reward DESC, tokens ASC)
        save_block_best_example(block_id, reward=best["reward"], tokens=best["tokens"], path=best.get("path", []))

    else:
        # ======== EXAMPLES-ONLY branch (NO SEARCH) ========
        print("Running EXAMPLES-ONLY (no search/MCTS).")
        exemplar = load_block_best_example(block_id)  # single exemplar or []
        composed_prompt = build_examples_only_prompt(problem_description, exemplar)

        try:
            run_start_time = time.time()
            total_tokens = 0
            # One-shot code generation (no sentences/UCT)
            code, token_used = llm_utils.generate_code_from_prompt(client, composed_prompt, previous_code="")
            total_tokens += token_used
            full_code = code

            reward, explanation, success, token_used, constraint_satisfied, matches_ground_truth = llm_utils.get_final_reward_from_output(
                full_code, problem_description, client, ground_truth=ground_truth, tolerance=tolerance
            )
            total_tokens += token_used
            reward_score = float(reward)
            print(explanation, reward_score)

            # Self-refine
            attempt = 0
            while (not success or reward_score < REWARD_THRESHOLD) and attempt < SELF_REFINE_MAX_RETRIES:
                print(f"\n🔁 Self-Refine Attempt {attempt + 1} (EXAMPLES-ONLY)")
                revised_code, token_used = llm_utils.revise_code_based_on_feedback(client, full_code, explanation, problem_description)
                total_tokens += token_used
                reward, explanation, success, token_used, constraint_satisfied, matches_ground_truth = llm_utils.get_final_reward_from_output(
                    revised_code, problem_description, client, ground_truth=ground_truth, tolerance=tolerance
                )
                total_tokens += token_used
                reward_score = float(reward)
                print(f"Tokens used: {total_tokens}, Reward: {reward_score}")
                full_code = revised_code
                attempt += 1

            if success:
                success_count += 1
                results.append(reward_score)
            else:
                results.append(-1.0)
            if constraint_satisfied:
                constraint_satisfied_count += 1
            if matches_ground_truth:
                ground_truth_match_count += 1

            run_time = time.time() - run_start_time
            time_counts.append(run_time)

            per_problem_candidates = [{
                "run": 1,
                "reward": reward_score,
                "success": bool(success),
                "constraint_satisfied": bool(constraint_satisfied),
                "matches_ground_truth": bool(matches_ground_truth),
                "tokens": total_tokens,
                "time": run_time,
                "code": full_code
            }]
            token_counts.append(total_tokens)

        except Exception as e:
            print(f"Examples-only run failed with error: {e}")
            results.append(-1.0)
            run_time = time.time() - run_start_time
            time_counts.append(run_time)

    successful = [c for c in per_problem_candidates if c["success"]]
    best = max(successful, key=lambda c: c["reward"]) if successful else max(per_problem_candidates, key=lambda c: c["reward"])

    avg_score = float(best["reward"])
    std_score = 0.0
    succ_rate = 1.0 if best["success"] else 0.0
    avg_tokens = float(best["tokens"])
    avg_time = float(best.get("time", 0.0))
    constraint_satisfied_rate = 1.0 if best["constraint_satisfied"] else 0.0
    ground_truth_match_rate = 1.0 if best["matches_ground_truth"] else 0.0

    print("\n--- Per-Problem Summary ---")
    print(f"Scores: {results} | mean={avg_score:.2f} std={std_score:.2f}")
    print(f"Success Rate: {succ_rate:.2%}")
    print(f"Avg Token Usage: {avg_tokens:.0f}")
    print(f"Avg Time: {avg_time:.2f}s")
    print(f"Constraint Satisfaction Rate: {constraint_satisfied_rate:.2%}")
    print(f"Ground Truth Match Rate: {ground_truth_match_rate:.2%}")

    # Persist per-problem summary
    summary_row = {
        "problem_index": idx,
        "condition": "SEARCH" if use_search else "EXAMPLES_ONLY",
        "avg_score": avg_score,
        "std_score": std_score,
        "success_rate": succ_rate,
        "avg_tokens": avg_tokens,
        "avg_time": avg_time,
        "constraint_satisfaction_rate": constraint_satisfied_rate,
        "ground_truth_match_rate": ground_truth_match_rate
    }
    all_results.append(summary_row)
    all_tokens.append(avg_tokens)

    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4)

    print(f"✅ Saved problem {idx} summary to {RESULTS_JSON}")

# =========================
# Global summary
# =========================
print("\n=== Overall Summary ===")
print(f"Mean success rate (per-problem):   {np.mean([r['success_rate'] for r in all_results]):.2%}")
print(f"Mean score (per-problem avg):      {np.mean([r['avg_score'] for r in all_results]):.2f}")
print(f"SD of per-problem avg scores:      {np.std([r['avg_score'] for r in all_results]):.2f}")
print(f"Mean token usage:                  {np.mean(all_tokens):.0f}")
print(f"Mean time per problem:             {np.mean([r['avg_time'] for r in all_results]):.2f}s")
print(f"Mean constraint satisfaction rate: {np.mean([r['constraint_satisfaction_rate'] for r in all_results]):.2%}")
print(f"Mean ground truth match rate:      {np.mean([r['ground_truth_match_rate'] for r in all_results]):.2%}")
