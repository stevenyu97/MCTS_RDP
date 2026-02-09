import os
import json
import time
import tempfile
import subprocess
import numpy as np
from openai import OpenAI
import llm_utils  # needs: clean_llm_generated_code, get_final_reward_from_output
import pandas as pd

# ======================
# Config
# ======================
MODEL = "gpt-5"
#TEMPERATURE = 0.3
TEMPERATURE = 1
TIMEOUT_SEC = 30
#DATASET_PATH = "all_questions.jsonl"
DATASET_PATH = "/home/ubuntu/LLM_Planning/problems_100.jsonl"
NUM_PROBLEMS = 100         # slice first K problems
RUNS_PER_PROBLEM = 1
TOLERANCE = 1e-4

client = OpenAI(api_key="")

# ======================
# Helpers
# ======================
def generate_code_oneshot(problem_description: str):
    """
    One-shot baseline: ask LLM for a single runnable CVXPY script.
    Instructs model to print a final scalar when appropriate using 'FINAL_ANSWER: <number>'.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are a Python optimization expert. Solve the following optimization problem using any suitable Python library.\n"
                "Generate complete, runnable code that defines variables, constraints, and objective, executes the optimization, and prints the final result.\n"
                "If a single numeric value is requested, also print:\n"
                "  FINAL_ANSWER: <number>\n"
                "Do not include explanations or comments."
            )
        },
        {
            "role": "user",
            "content": f"Problem: {problem_description}\n\nGenerate the complete Python code now."
        }
    ]
    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=TEMPERATURE
    )
    token_used = resp.usage.total_tokens
    code_raw = resp.choices[0].message.content.strip()
    return llm_utils.clean_llm_generated_code(code_raw), token_used

def run_code_string(code_string: str, timeout_s: int = TIMEOUT_SEC):
    """
    Execute code without persisting to disk (uses a temp file; auto removed).
    Returns (stdout+stderr, executed_ok).
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp:
        tmp.write(code_string)
        tmp_path = tmp.name
    try:
        result = subprocess.run(
            ["python3", tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout_s
        )
        return result.stdout + "\n" + result.stderr, (result.returncode == 0)
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

# ======================
# Load problems
# ======================
problems = []


with open(DATASET_PATH, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        problems.append({
            "problem_description": obj.get("en_question", ""),
            "ground_truth": obj.get("en_answer", "")
        })



all_results = []
all_tokens  = []
results_path = "baseline_aq.json"

# Ensure file exists before starting
if not os.path.exists(results_path):
    with open(results_path, "w") as f:
        json.dump({"results": []}, f)
# ======================
# Main loop
# ======================
for idx, item in enumerate(problems, start=1):
    print(f"\n================ Problem {idx}/{len(problems)} ================")
    problem_description = item["problem_description"]
    gt_str = item["ground_truth"]

    try:
        ground_truth = float(gt_str) if str(gt_str).strip() != "" else None
    except Exception:
        ground_truth = None

    results = []
    success_count = 0
    token_counts = []
    time_counts = []
    constraint_satisfied_count = 0
    ground_truth_match_count = 0

    for run in range(RUNS_PER_PROBLEM):
        print(f"\n--- Baseline run {run + 1}/{RUNS_PER_PROBLEM} ---")
        run_start_time = time.time()
        total_tokens = 0

        # 1) Generate code (one-shot)
        code, tok_gen = generate_code_oneshot(problem_description)
        total_tokens += tok_gen

        # 2) Execute
        output, executed_ok = run_code_string(code, timeout_s=TIMEOUT_SEC)
        print(output)
        print("Executed:", executed_ok)

        # get_final_reward_from_output must return:
        #  (score, explanation, executed_successfully, token_used, constraint_satisfied, matches_ground_truth)
        score, explanation, executed_successfully, tok_eval, constraint_satisfied, matches_gt = (
            llm_utils.get_final_reward_from_output(
                code_string=code,
                problem_description=problem_description,
                client=client,
                ground_truth=ground_truth,
                tolerance=TOLERANCE
            )
        )
        total_tokens += tok_eval

        print(f"Score: {score:.2f}")
        print("Constraint satisfied:", constraint_satisfied)
        print("Matches ground truth:", matches_gt)
        print(f"Tokens (this run): {total_tokens}")
        
        run_time = time.time() - run_start_time
        time_counts.append(run_time)
        print(f"Time (this run): {run_time:.2f}s")

        token_counts.append(total_tokens)
        results.append(score)
        if executed_ok and executed_successfully:
            success_count += 1
        if constraint_satisfied:
            constraint_satisfied_count += 1
        if matches_gt:
            ground_truth_match_count += 1

    # Per-problem summary
    avg_score = float(np.mean(results))
    std_score = float(np.std(results))
    succ_rate = success_count / max(1, len(results))
    avg_tokens = float(np.mean(token_counts))
    avg_time = float(np.mean(time_counts))
    constraint_satisfied_rate = constraint_satisfied_count / max(1, len(results))
    ground_truth_match_rate = ground_truth_match_count / max(1, len(results))

    print("\n--- Per-Problem Summary ---")
    print(f"Scores: {results} | mean={avg_score:.2f} std={std_score:.2f}")
    print(f"Success Rate: {succ_rate:.2%}")
    print(f"Avg Token Usage: {avg_tokens:.0f}")
    print(f"Avg Time: {avg_time:.2f}s")
    print(f"Constraint Satisfaction Rate: {constraint_satisfied_rate:.2%}")
    print(f"Ground Truth Match Rate: {ground_truth_match_rate:.2%}")

    all_results.append({
        "problem_index": idx,
        "avg_score": avg_score,
        "std_score": std_score,
        "success_rate": succ_rate,
        "avg_tokens": avg_tokens,
        "avg_time": avg_time,
        "constraint_satisfaction_rate": constraint_satisfied_rate,
        "ground_truth_match_rate": ground_truth_match_rate
    })
    all_tokens.append(avg_tokens)
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=4)

    print(f"✅ Saved iteration {idx} results to {results_path}")




# Global summary
print("\n=== Overall Summary ===")
print(f"Mean of per-problem success rates: {np.mean([r['success_rate'] for r in all_results]):.2%}")
print(f"Mean of per-problem avg scores:    {np.mean([r['avg_score'] for r in all_results]):.2f}")
print(f"sd of per-problem avg scores:    {np.std([r['avg_score'] for r in all_results]):.2f}")
print(f"Mean token usage:                   {np.mean(all_tokens):.0f}")
print(f"Mean time per problem:              {np.mean([r['avg_time'] for r in all_results]):.2f}s")
print(f"Mean constraint satisfaction rate:  {np.mean([r['constraint_satisfaction_rate'] for r in all_results]):.2%}")
print(f"Mean ground truth match rate:       {np.mean([r['ground_truth_match_rate'] for r in all_results]):.2%}")
