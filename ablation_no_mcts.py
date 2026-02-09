"""
Ablation Study: Decomposition WITHOUT MCTS
Tests whether MCTS tree search and UCT selection help performance.

Key change: Keep sentence decomposition but remove MCTS tree navigation.
Generate prompts sequentially for each sentence without tree search or backpropagation.
"""

import json
import os
import time
from typing import Optional
import numpy as np
from pathlib import Path
from openai import OpenAI

import llm_utils
from typing import List, Dict, Optional, Any


# =========================
# Config
# =========================
DATASET_PATH = "/home/ubuntu/LLM_Planning/all_questions.jsonl"
#DATASET_PATH = "/home/ubuntu/LLM_Planning/problems_100.jsonl"

RESULTS_JSON = "ablation_no_mcts_results.json"
SELF_REFINE_MAX_RETRIES = 2
REWARD_THRESHOLD = 1

client = OpenAI(api_key=)

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


    results = []
    success_count = 0
    token_counts = []
    time_counts = []
    constraint_satisfied_count = 0
    ground_truth_match_count = 0

    # ======== DECOMPOSITION WITHOUT MCTS (single run per problem) ========
    # Run decomposition on EVERY problem independently
    print("Running DECOMPOSITION mode (NO MCTS TREE)")

    try:
        print(f"\n=== Single Run (Sequential Decomposition) ===")
        run_start_time = time.time()
        total_tokens = 0

        # Step 1: Decompose problem into sentences (KEEP THIS)
        sentences, token_used = llm_utils.generate_structured_sentences(client, problem_description)
        total_tokens += token_used
        print(f"Decomposed into {len(sentences)} sentences")

        # Step 2: Process each sentence sequentially WITHOUT MCTS
        cumulative_code = ""
        prompt_score_path = []
        for depth, sentence in enumerate(sentences):
            is_last = (depth == len(sentences) - 1)
            
            # Generate prompt for this sentence (no tree search, just sequential)
            prompt, token_used = llm_utils.select_prompt_from_llm(
                client, sentence, depth=depth, is_last=is_last
            )
            total_tokens += token_used
            
            # Score the prompt (for consistency with original, but not used for selection)
            score, token_used = llm_utils.score_prompt(client, prompt)
            total_tokens += token_used
            score = float(score)
            
            print(f"Depth {depth} | Score: {score:.2f} | Prompt: {prompt[:80]}...")
            
            # Generate code incrementally
            code, token_used = llm_utils.generate_code_from_prompt(
                client, prompt, previous_code=cumulative_code
            )
            total_tokens += token_used
            cumulative_code = code
            prompt_score_path.append((prompt, score))

        # Step 3: Evaluate
        full_code = "# === Auto-generated Optimization Script ===\n\n" + cumulative_code
        reward, explanation, success, token_used, constraint_satisfied, matches_ground_truth = llm_utils.get_final_reward_from_output(
            full_code, problem_description, client, ground_truth=ground_truth, tolerance=tolerance
        )
        total_tokens += token_used
        reward_score = float(reward)
        print(f"Initial reward: {reward_score:.2f}, Success: {success}")

        # Step 4: Self-refine (NO MCTS BACKPROPAGATION)
        attempt = 0
        while (not success or reward_score < REWARD_THRESHOLD) and attempt < SELF_REFINE_MAX_RETRIES:
            print(f"\n🔁 Self-Refine Attempt {attempt + 1}")
            revised_code, token_used = llm_utils.revise_code_based_on_feedback(
                client, full_code, explanation, problem_description
            )
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
        token_counts.append(total_tokens)

    except Exception as e:
        print(f"Run failed with error: {e}")
        results.append(-1.0)
        run_time = time.time() - run_start_time
        time_counts.append(run_time)
        # Use defaults for failed run
        reward_score = -1.0
        success = False
        constraint_satisfied = False
        matches_ground_truth = False
        total_tokens = 0
        token_used = 0

    avg_score = float(reward_score)
    std_score = 0.0
    succ_rate = 1.0 if success else 0.0
    avg_tokens = float(total_tokens)
    avg_time = float(run_time)
    constraint_satisfied_rate = 1.0 if constraint_satisfied else 0.0
    ground_truth_match_rate = 1.0 if matches_ground_truth else 0.0

    print("\n--- Per-Problem Summary (No MCTS) ---")
    print(f"Scores: {results} | mean={avg_score:.2f} std={std_score:.2f}")
    print(f"Success Rate: {succ_rate:.2%}")
    print(f"Avg Token Usage: {avg_tokens:.0f}")
    print(f"Avg Time: {avg_time:.2f}s")
    print(f"Constraint Satisfaction Rate: {constraint_satisfied_rate:.2%}")
    print(f"Ground Truth Match Rate: {ground_truth_match_rate:.2%}")

    summary_row = {
        "problem_index": idx,
        "condition": "DECOMP_NO_MCTS",
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
print("\n=== Overall Summary (Ablation: Decomposition WITHOUT MCTS) ===")
print(f"Mean success rate (per-problem):   {np.mean([r['success_rate'] for r in all_results]):.2%}")
print(f"Mean score (per-problem avg):      {np.mean([r['avg_score'] for r in all_results]):.2f}")
print(f"SD of per-problem avg scores:      {np.std([r['avg_score'] for r in all_results]):.2f}")
print(f"Mean token usage:                  {np.mean(all_tokens):.0f}")
print(f"Mean time per problem:             {np.mean([r['avg_time'] for r in all_results]):.2f}s")
print(f"Mean constraint satisfaction rate: {np.mean([r['constraint_satisfaction_rate'] for r in all_results]):.2%}")
print(f"Mean ground truth match rate:      {np.mean([r['ground_truth_match_rate'] for r in all_results]):.2%}")
