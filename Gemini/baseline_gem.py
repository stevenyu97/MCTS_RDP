import os
import json
import tempfile
import subprocess
import time
import numpy as np
import google.generativeai as genai
print(f"Current version: {genai.__version__}")
import llm_gemutils  
import pandas as pd

# ======================
# Config
# ======================

MODEL = "gemini-2.5-pro"
TEMPERATURE = 1.0
TIMEOUT_SEC = 30
DATASET_PATH = "/home/ubuntu/LLM_Planning/all_questions.jsonl"
#DATASET_PATH = "/home/ubuntu/LLM_Planning/problems_100.jsonl"
NUM_PROBLEMS = 242
RUNS_PER_PROBLEM = 1
TOLERANCE = 1e-4

# ======================
# Authenticate Gemini
# ======================
genai.configure(api_key="")

# ======================
# Helpers
# ======================
def generate_code_oneshot(problem_description: str):
    system_instruction = (
        "You are a Python optimization expert. Solve the following optimization problem using any suitable Python library.\n"
        "Generate complete, runnable code that defines variables, constraints, and objective, executes the optimization, and prints the final result.\n"
        "If a single numeric value is requested, also print:\n"
        "  FINAL_ANSWER: <number>\n"
        "Do not include explanations or comments."
    )
    
    prompt = f"Problem: {problem_description}\n\nGenerate the complete Python code now."

    try:
        # Initialize model
        model = genai.GenerativeModel(
            model_name=MODEL,
            system_instruction=system_instruction
        )


        safety_settings = {
            genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
            genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
            genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: genai.types.HarmBlockThreshold.BLOCK_NONE,
            genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: genai.types.HarmBlockThreshold.BLOCK_NONE,
        }

        response = model.generate_content(
            prompt,
            generation_config={"temperature": TEMPERATURE},
            safety_settings=safety_settings
        )
        
        code_raw = response.text.strip()
        token_count = response.usage_metadata.total_token_count
        return llm_gemutils.clean_llm_generated_code(code_raw), token_count

    except Exception as e:
        print(f"Error generating code: {e}")
        return f"# Error: {str(e)}", 0

def run_code_string(code_string: str, timeout_s: int = TIMEOUT_SEC):
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
    except subprocess.TimeoutExpired:
        return "Execution Timed Out", False
    except Exception as e:
        return f"Execution Error: {e}", False
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass

# ======================
# Load problems
# ======================
problems = []
if not os.path.exists(DATASET_PATH):
    print(f"Warning: {DATASET_PATH} not found. Creating dummy data for testing.")
    problems = [{"problem_description": "Minimize x^2 subject to x >= 2", "ground_truth": "4"}]
else:
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
all_tokens = []
all_times = []
results_path = "baseline_aq.json"

if not os.path.exists(results_path):
    with open(results_path, "w") as f:
        json.dump({"results": []}, f)

# ======================
# Main loop
# ======================
print(f"Starting evaluation on {len(problems)} problems using {MODEL}...")

for idx, item in enumerate(problems[:NUM_PROBLEMS], start=1):
    print(f"\n================ Problem {idx}/{min(len(problems), NUM_PROBLEMS)} ================")
    problem_description = item["problem_description"]
    gt_str = item["ground_truth"]

    try:
        ground_truth = float(gt_str) if str(gt_str).strip() != "" else None
    except Exception:
        ground_truth = None

    results = []
    success_count = 0
    token_counts = []
    run_times = []
    constraint_satisfied_count = 0
    ground_truth_match_count = 0

    for run in range(RUNS_PER_PROBLEM):
        print(f"\n--- Baseline run {run + 1}/{RUNS_PER_PROBLEM} ---")
        start_time = time.time()

        # 1. Generate Code
        code, tok_gen = generate_code_oneshot(problem_description)
        #print(code)
        
        # 2. Run Code
        output, executed_ok = run_code_string(code, timeout_s=TIMEOUT_SEC)
        print(f"Output len: {len(output)} chars")
        print("Executed:", executed_ok)

        # 3. Evaluate (Using the imported gemutils helper)
        score, explanation, executed_successfully, tok_eval, constraint_satisfied, matches_gt = (
            llm_gemutils.get_final_reward_from_output(
                code_string=code,
                problem_description=problem_description,
                ground_truth=ground_truth,
                tolerance=TOLERANCE
            )
        )
        
        total_tokens = tok_gen + tok_eval

        print(f"Score: {score:.2f}")
        print("Constraint satisfied:", constraint_satisfied)
        print("Matches ground truth:", matches_gt)
        print(f"Tokens (this run): {total_tokens}")

        token_counts.append(total_tokens)
        results.append(score)
        
        if executed_successfully: 
            success_count += 1
        if constraint_satisfied:
            constraint_satisfied_count += 1
        if matches_gt:
            ground_truth_match_count += 1

        run_times.append(time.time() - start_time)

    avg_score = float(np.mean(results)) if results else 0.0
    std_score = float(np.std(results)) if results else 0.0
    succ_rate = success_count / max(1, RUNS_PER_PROBLEM)
    avg_tokens = float(np.mean(token_counts)) if token_counts else 0.0
    avg_time = float(np.mean(run_times)) if run_times else 0.0
    constraint_satisfied_rate = constraint_satisfied_count / max(1, RUNS_PER_PROBLEM)
    ground_truth_match_rate = ground_truth_match_count / max(1, RUNS_PER_PROBLEM)

    print("\n--- Per-Problem Summary ---")
    print(f"Scores: {results} | mean={avg_score:.2f} std={std_score:.2f}")
    print(f"Success Rate: {succ_rate:.2%}")
    print(f"Avg Time: {avg_time:.2f} sec")

    all_results.append({
        "problem_index": idx,
        "avg_score": avg_score,
        "std_score": std_score,
        "success_rate": succ_rate,
        "avg_tokens": avg_tokens,
        "avg_time_sec": avg_time,
        "constraint_satisfaction_rate": constraint_satisfied_rate,
        "ground_truth_match_rate": ground_truth_match_rate
    })
    all_tokens.append(avg_tokens)
    all_times.append(avg_time)
    
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=4)
    print(f"✅ Saved iteration {idx} results to {results_path}")

# ======================
# Final Summary
# ======================
if all_results:
    print("\n=== Overall Summary ===")
    print(f"Mean of per-problem success rates: {np.mean([r['success_rate'] for r in all_results]):.2%}")
    print(f"Mean of per-problem avg scores:    {np.mean([r['avg_score'] for r in all_results]):.2f}")
    print(f"sd of per-problem avg scores:      {np.std([r['avg_score'] for r in all_results]):.2f}")
    print(f"Mean token usage:                  {np.mean(all_tokens):.0f}")
    print(f"Mean time per problem (s):         {np.mean(all_times):.2f}")
    print(f"Mean constraint satisfaction rate: {np.mean([r['constraint_satisfaction_rate'] for r in all_results]):.2%}")
    print(f"Mean ground truth match rate:      {np.mean([r['ground_truth_match_rate'] for r in all_results]):.2%}")
else:
    print("\nNo results to summarize.")