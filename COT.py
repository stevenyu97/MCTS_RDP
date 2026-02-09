import os
import json
import time
import numpy as np
from openai import OpenAI

import llm_utils  

# -------------------------
# Config
# -------------------------
#DATASET_PATH = "all_questions.jsonl"
DATASET_PATH = "/home/ubuntu/LLM_Planning/problems_100.jsonl"

NUM_TRIALS_PER_PROBLEM = 1
GROUND_TRUTH_TOLERANCE = 1e-4

client = OpenAI(api_key="")


def cot_generate_code_for_problem(problem_description: str):
    """
    Chain-of-Thought style pipeline using your llm_utils:
      1) generate_structured_sentences
      2) for each sentence: select_prompt_from_llm -> generate_code_from_prompt (with cumulative context)
    Returns: (full_code, total_tokens_used)
    """
    total_tokens = 0
    # 1) decomposition
    sentences, tok = llm_utils.generate_structured_sentences(client, problem_description)
    total_tokens += tok

    # 2) roll forward with cumulative context
    cumulative_code = ""
    for i,s in enumerate(sentences):
        is_last = (i == len(sentences) - 1)
        prompt, tok = llm_utils.select_prompt_from_llm(client, s, i, is_last)
        total_tokens += tok


        code, tok = llm_utils.generate_code_from_prompt(client, prompt, previous_code=cumulative_code)
        total_tokens += tok

        # In your pipeline, "new code replaces old"
        cumulative_code = code

    full_code = "# === Auto-generated Optimization Script (CoT Baseline) ===\n\n" + cumulative_code
    return full_code, total_tokens


def evaluate_with_grader(code_str, problem_description, ground_truth, tolerance):

    out = llm_utils.get_final_reward_from_output(
        code_string=code_str,
        problem_description=problem_description,
        client=client,
        ground_truth=ground_truth,
        tolerance=tolerance
    )

    # Backward-compatible unpacking
    if len(out) == 6:
        score, explanation, executed, token_used, constraint_satisfied, matches_ground_truth = out
    else:

        score, explanation, executed, token_used = out
        constraint_satisfied = False
        matches_ground_truth = False

    return float(score), str(explanation), bool(executed), int(token_used), bool(constraint_satisfied), bool(matches_ground_truth)


# -------------------------
# Load problems
# -------------------------
problems = []
with open(DATASET_PATH, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        problems.append({
            "problem_description": obj["en_question"],
            "ground_truth": str(obj.get("en_answer", "")).strip()
        })





all_results = []
all_tokens = []

# -------------------------
# Main loop: per-problem evaluation
# -------------------------
for idx, item in enumerate(problems, start=1):
    print(f"\n================ Problem {idx}/{len(problems)} ================")
    problem_description = item["problem_description"]
    ground_truth = item["ground_truth"]

    # Per-problem tallies
    scores = []
    token_counts = []
    time_counts = []
    success_count = 0
    constraint_satisfied_count = 0
    ground_truth_match_count = 0

    for trial in range(NUM_TRIALS_PER_PROBLEM):
        print(f"\n-- Trial {trial+1}/{NUM_TRIALS_PER_PROBLEM} --")
        trial_start_time = time.time()

        # 1) CoT pipeline to produce a full script
        code_str, gen_tokens = cot_generate_code_for_problem(problem_description)

        # 2) Grade using your JSON grader (uniform across methods)
        score, explanation, executed, eval_tokens, constraint_ok, matches_gt = evaluate_with_grader(
            code_str, problem_description, ground_truth, GROUND_TRUTH_TOLERANCE
        )

        total_tokens = gen_tokens + eval_tokens
        token_counts.append(total_tokens)
        scores.append(score)
        if executed:
            success_count += 1
        if constraint_ok:
            constraint_satisfied_count += 1
        if matches_gt:
            ground_truth_match_count += 1
        
        trial_time = time.time() - trial_start_time
        time_counts.append(trial_time)

        print(f"Score: {score:.2f} | Executed: {executed} | "
              f"Constraint OK: {constraint_ok} | GT match: {matches_gt} | "
              f"Tokens: {total_tokens} | Time: {trial_time:.2f}s")

    # Per-problem summary
    avg_score = float(np.mean(scores))
    std_score = float(np.std(scores))
    succ_rate = success_count / max(1, len(scores))
    avg_tokens = float(np.mean(token_counts))
    avg_time = float(np.mean(time_counts))
    constraint_satisfaction_rate = constraint_satisfied_count / max(1, len(scores))
    ground_truth_match_rate = ground_truth_match_count / max(1, len(scores))

    print("\n--- Per-Problem Summary (CoT Baseline) ---")
    print(f"Scores: {scores} | mean={avg_score:.2f} std={std_score:.2f}")
    print(f"Success Rate: {succ_rate:.2%}")
    print(f"Avg Token Usage: {avg_tokens:.0f}")
    print(f"Avg Time: {avg_time:.2f}s")
    print(f"Constraint Satisfaction Rate: {constraint_satisfaction_rate:.2%}")
    print(f"Ground Truth Match Rate: {ground_truth_match_rate:.2%}")

    all_results.append({
        "problem_index": idx,
        "avg_score": avg_score,
        "std_score": std_score,
        "success_rate": succ_rate,
        "avg_tokens": avg_tokens,
        "avg_time": avg_time,
        "constraint_satisfaction_rate": constraint_satisfaction_rate,
        "ground_truth_match_rate": ground_truth_match_rate
    })
    all_tokens.append(avg_tokens)

# -------------------------
# Overall summary across problems
# -------------------------
print("\n=== Overall Summary (CoT Baseline) ===")
print(f"Mean of per-problem success rates: {np.mean([r['success_rate'] for r in all_results]):.2%}")
print(f"Mean of per-problem avg scores:    {np.mean([r['avg_score'] for r in all_results]):.2f}")
print(f"sd of per-problem avg scores:    {np.std([r['avg_score'] for r in all_results]):.2f}")
print(f"Mean token usage:                   {np.mean(all_tokens):.0f}")
print(f"Mean time per problem:              {np.mean([r['avg_time'] for r in all_results]):.2f}s")
print(f"Mean constraint satisfaction rate:  {np.mean([r['constraint_satisfaction_rate'] for r in all_results]):.2%}")
print(f"Mean ground truth match rate:       {np.mean([r['ground_truth_match_rate'] for r in all_results]):.2%}")
