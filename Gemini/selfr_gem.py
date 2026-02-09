import os
import json
import time
import numpy as np

import llm_gemutils

# -------------------------
# Config
# -------------------------
DATASET_PATH = "/home/ubuntu/LLM_Planning/all_questions.jsonl"
#DATASET_PATH = "/home/ubuntu/LLM_Planning/problems_100.jsonl"

NUM_TRIALS_PER_PROBLEM = 1
MAX_REFINES = 2
REWARD_THRESHOLD = 1.0
GROUND_TRUTH_TOLERANCE = 1e-4



def self_refine_generate_initial_code(problem: str):
    """
    One-shot code generation (no decomposition), converted to use llm_utils.call_gemini.
    """
    system_instruction = (
        "You are a Python optimization expert. Solve the following optimization problem using any suitable Python library.\n"
        "Generate complete, runnable code that defines variables, constraints, and objective, executes the optimization, and prints the final result.\n"
        "If a single numeric value is requested, also print:\n"
        "  FINAL_ANSWER: <number>\n"
        "Do not include explanations or comments."
    )
    
    user_prompt = f"Problem: {problem}\n\nGenerate complete Python code to solve it. No comments."


    code_raw, tokens_used = llm_gemutils.call_gemini(
        system_instruction=system_instruction,
        user_prompt=user_prompt,
        temperature=1.0  
    )

    return llm_gemutils.clean_llm_generated_code(code_raw), tokens_used


def evaluate_with_grader(code_str, problem_description, ground_truth, tolerance):
    """
    Your updated grader is expected to return:
      (score, explanation, executed, token_used, constraint_satisfied, matches_ground_truth)
    We explicitly remove the client argument from the call.
    """
    out = llm_gemutils.get_final_reward_from_output(
        code_string=code_str,
        problem_description=problem_description,
        ground_truth=ground_truth,
        tolerance=tolerance
    )

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
all_times = []

# -------------------------
# Main loop (per-problem)
# -------------------------
for idx, item in enumerate(problems, start=1):
    print(f"\n================ Problem {idx}/{len(problems)} ================")
    problem_description = item["problem_description"]
    ground_truth = item["ground_truth"]

    scores = []
    token_counts = []
    run_times = []
    success_count = 0
    constraint_satisfied_count = 0
    ground_truth_match_count = 0

    for trial in range(NUM_TRIALS_PER_PROBLEM):
        print(f"\n-- Trial {trial+1}/{NUM_TRIALS_PER_PROBLEM} --")
        start_time = time.time()

        # 1) Initial one-shot code
        code_str, gen_tokens = self_refine_generate_initial_code(problem_description)
        #print("Generated Code:\n", code_str)

        # 2) Evaluate
        score, explanation, executed, eval_tokens, constraint_ok, matches_gt = evaluate_with_grader(
            code_str, problem_description, ground_truth, GROUND_TRUTH_TOLERANCE
        )
        total_tokens = gen_tokens + eval_tokens

        # 3) Self-refine loop if needed
        refines = 0
        while refines < MAX_REFINES and (not executed or score < REWARD_THRESHOLD):
            code_str, fb_tokens = llm_gemutils.revise_code_based_on_feedback(
                code=code_str,
                feedback=explanation,
                problem_description=problem_description
            )
            total_tokens += fb_tokens

            # Re-evaluate
            score, explanation, executed, eval_tokens, constraint_ok, matches_gt = evaluate_with_grader(
                code_str, problem_description, ground_truth, GROUND_TRUTH_TOLERANCE
            )
            total_tokens += eval_tokens
            refines += 1

        # record trial stats
        scores.append(score)
        token_counts.append(total_tokens)
        if executed:
            success_count += 1
        if constraint_ok:
            constraint_satisfied_count += 1
        if matches_gt:
            ground_truth_match_count += 1

        run_times.append(time.time() - start_time)

        print(f"Score: {score:.2f} | Executed: {executed} | "
              f"Constraint OK: {constraint_ok} | GT match: {matches_gt} | "
              f"Tokens: {total_tokens}")

    # Per-problem summary
    avg_score = float(np.mean(scores))
    std_score = float(np.std(scores))
    succ_rate = success_count / max(1, len(scores))
    avg_tokens = float(np.mean(token_counts))
    avg_time = float(np.mean(run_times)) if run_times else 0.0
    constraint_satisfaction_rate = constraint_satisfied_count / max(1, len(scores))
    ground_truth_match_rate = ground_truth_match_count / max(1, len(scores))

    print("\n--- Per-Problem Summary (Self-Refine Baseline) ---")
    print(f"Scores: {scores} | mean={avg_score:.2f} std={std_score:.2f}")
    print(f"Success Rate: {succ_rate:.2%}")
    print(f"Avg Token Usage: {avg_tokens:.0f}")
    print(f"Avg Time: {avg_time:.2f} sec")
    print(f"Constraint Satisfaction Rate: {constraint_satisfaction_rate:.2%}")
    print(f"Ground Truth Match Rate: {ground_truth_match_rate:.2%}")

    all_results.append({
        "problem_index": idx,
        "avg_score": avg_score,
        "std_score": std_score,
        "success_rate": succ_rate,
        "avg_tokens": avg_tokens,
        "avg_time_sec": avg_time,
        "constraint_satisfaction_rate": constraint_satisfaction_rate,
        "ground_truth_match_rate": ground_truth_match_rate
    })
    all_tokens.append(avg_tokens)
    all_times.append(avg_time)

# -------------------------
# Overall summary
# -------------------------
print("\n=== Overall Summary (Self-Refine Baseline) ===")
print(f"Mean of per-problem success rates: {np.mean([r['success_rate'] for r in all_results]):.2%}")
print(f"Mean of per-problem avg scores:    {np.mean([r['avg_score'] for r in all_results]):.2f}")
print(f"Mean token usage:                  {np.mean(all_tokens):.0f}")
print(f"sd of per-problem avg scores:    {np.std([r['avg_score'] for r in all_results]):.2f}")
print(f"Mean time per problem (s):         {np.mean(all_times):.2f}")
print(f"Mean constraint satisfaction rate:  {np.mean([r['constraint_satisfaction_rate'] for r in all_results]):.2%}")
print(f"Mean ground truth match rate:      {np.mean([r['ground_truth_match_rate'] for r in all_results]):.2%}")