import os
import sys
import json
import time
import tempfile
import shutil
import subprocess
import numpy as np
import google.generativeai as genai
import llm_gemutils

# Configure Gemini API
genai.configure(api_key="")

LIC_PATH = os.path.join(os.path.dirname(__file__), "..", "optimus_gem", "gurobi.lic")
if os.path.exists(LIC_PATH):
    os.environ["GRB_LICENSE_FILE"] = LIC_PATH

# ======================
# Config  
# ======================
#DATASET_PATH = "/home/ubuntu/LLM_Planning/problems_100.jsonl"
DATASET_PATH = "/home/ubuntu/LLM_Planning/all_questions.jsonl"

NUM_TRIALS_PER_PROBLEM = 1
GROUND_TRUTH_TOLERANCE = 1e-4




def run_optimus_on_problem(problem_description: str, work_dir: str):
    """
    Run OptiMUS on a single problem by calling it as a subprocess.
    Returns: (generated_code, success, optimus_token_total)
    """
    try:
        # Create required directory structure
        os.makedirs(work_dir, exist_ok=True)
        
        # Create desc.txt with problem description
        with open(os.path.join(work_dir, "desc.txt"), "w") as f:
            f.write(problem_description)
        
        # Create empty params.json
        with open(os.path.join(work_dir, "params.json"), "w") as f:
            json.dump({}, f)
        
        # Create empty labels.json  
        with open(os.path.join(work_dir, "labels.json"), "w") as f:
            json.dump({}, f)
        
        # Create empty data.json (required by generated code)
        with open(os.path.join(work_dir, "data.json"), "w") as f:
            json.dump({}, f)
        
        # Run optimus main.py using absolute path
        base_dir = os.path.abspath(os.path.dirname(__file__))
        optimus_dir = os.path.join(base_dir, '..', 'optimus_gem')
        main_py_path = os.path.join(optimus_dir, 'main.py')
        env = dict(os.environ)
        if os.path.exists(LIC_PATH):
            env["GRB_LICENSE_FILE"] = LIC_PATH
        token_log_path = os.path.join(work_dir, "optimus_token_log.jsonl")
        env["OPTIMUS_TOKEN_LOG"] = token_log_path

        result = subprocess.run(
            [sys.executable, main_py_path, '--dir', work_dir, '--devmode', '1'],
            cwd=optimus_dir,
            capture_output=True,
            text=True,
            timeout=900,  # 15 minute timeout
            env=env,
        )
        
        # Check if code was generated
        code_path = os.path.join(work_dir, "run_dev", "code.py")
        data_json_path = os.path.join(work_dir, "run_dev", "data.json")
        optimus_tokens = 0
        if os.path.exists(token_log_path):
            try:
                with open(token_log_path, "r", encoding="utf-8") as tf:
                    for line in tf:
                        try:
                            obj = json.loads(line)
                            if "total_tokens" in obj:
                                optimus_tokens += obj.get("total_tokens", 0)
                            else:
                                optimus_tokens += obj.get("prompt_tokens", 0) + obj.get("completion_tokens", 0)
                        except Exception:
                            continue
            except Exception:
                pass

        if os.path.exists(code_path):
            with open(code_path, "r") as f:
                code = f.read()
            # Ensure the generated code loads data.json from the correct location when graded
            if os.path.exists(data_json_path):
                code = code.replace('with open("data.json", "r") as f:', f'with open(r"{data_json_path}", "r") as f:')
            return code, True, optimus_tokens
        else:
            print(f"OptiMUS failed: {result.stderr}")
            return "", False, optimus_tokens
            
    except Exception as e:
        print(f"OptiMUS pipeline error: {e}")
        return "", False, 0


def evaluate_with_grader(code_str, problem_description, ground_truth, tolerance):
    """
    Use the Gemini grader from llm_gemutils.
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
        # backward compatibility
        score, explanation, executed, token_used = out
        constraint_satisfied = False
        matches_ground_truth = False

    return float(score), str(explanation), bool(executed), int(token_used), bool(constraint_satisfied), bool(matches_ground_truth)


# -------------------------
# Load problems
# -------------------------
problems = []
with open(DATASET_PATH, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        problems.append({
            "problem_description": obj.get("en_question", ""),
            "ground_truth": str(obj.get("en_answer", "")).strip()
        })

# Optionally subset for testing
# problems = problems[:5]

all_results = []
all_tokens = []
all_times = []

# -------------------------
# Main loop
# -------------------------
for idx, item in enumerate(problems, start=1):
    print(f"\n================ Problem {idx}/{len(problems)} ================")
    problem_description = item["problem_description"]
    ground_truth = item["ground_truth"]

    scores = []
    token_counts = []
    time_counts = []
    explanations = []
    success_count = 0
    constraint_satisfied_count = 0
    ground_truth_match_count = 0

    for trial in range(NUM_TRIALS_PER_PROBLEM):
        print(f"\n-- Trial {trial+1}/{NUM_TRIALS_PER_PROBLEM} --")
        
        trial_start_time = time.time()
        
        # Create temporary working directory for this trial
        work_dir = tempfile.mkdtemp(prefix=f"optimus_p{idx}_t{trial}_")
        
        try:
            # Run OptiMUS pipeline
            code, optimus_success, optimus_tokens = run_optimus_on_problem(problem_description, work_dir)
            
            if not optimus_success or not code:
                print("OptiMUS failed to generate code")
                scores.append(-1.0)
                token_counts.append(optimus_tokens)
                explanations.append("OptiMUS failed to generate code")
                continue
            
            # Evaluate using Gemini grader
            score, explanation, executed, eval_tokens, constraint_ok, matches_gt = evaluate_with_grader(
                code, problem_description, ground_truth, GROUND_TRUTH_TOLERANCE
            )
            
            # Track total tokens: OptiMUS + evaluation
            total_tokens = optimus_tokens + eval_tokens  
            
            scores.append(score)
            token_counts.append(total_tokens)
            explanations.append(explanation)
            
            if executed and optimus_success:
                success_count += 1
            if constraint_ok:
                constraint_satisfied_count += 1
            if matches_gt:
                ground_truth_match_count += 1
            
            trial_time = time.time() - trial_start_time
            time_counts.append(trial_time)
            
            print(f"Score: {score:.2f} | Executed: {executed} | "
                f"Constraint OK: {constraint_ok} | GT match: {matches_gt} | "
                f"Tokens (OptiMUS+eval): {total_tokens} | Time: {trial_time:.2f}s")
            print(f"Explanation: {explanation}")
                  
        except Exception as e:
            print(f"Trial {trial+1} failed with error: {e}")
            scores.append(-1.0)
            token_counts.append(0)
            explanations.append(f"Trial failed with error: {str(e)}")
            trial_time = time.time() - trial_start_time
            time_counts.append(trial_time)
        finally:
            # Clean up temporary directory
            try:
                shutil.rmtree(work_dir)
            except Exception:
                pass

    # Per-problem summary
    avg_score = float(np.mean(scores))
    std_score = float(np.std(scores))
    succ_rate = success_count / max(1, len(scores))
    avg_tokens = float(np.mean(token_counts)) if token_counts else 0.0
    avg_time = float(np.mean(time_counts)) if time_counts else 0.0
    constraint_satisfaction_rate = constraint_satisfied_count / max(1, len(scores))
    ground_truth_match_rate = ground_truth_match_count / max(1, len(scores))

    print("\n--- Per-Problem Summary (OptiMUS + Gemini) ---")
    print(f"Scores: {scores} | mean={avg_score:.2f} std={std_score:.2f}")
    print(f"Success Rate: {succ_rate:.2%}")
    print(f"Avg Token Usage: {avg_tokens:.0f}")
    print(f"Avg Time: {avg_time:.2f}s")
    print(f"Constraint Satisfaction Rate: {constraint_satisfaction_rate:.2%}")
    print(f"Ground Truth Match Rate: {ground_truth_match_rate:.2%}")

    all_results.append({
        "problem_index": idx,
        "problem_description": problem_description,
        "ground_truth": ground_truth,
        "avg_score": avg_score,
        "std_score": std_score,
        "success_rate": succ_rate,
        "avg_tokens": avg_tokens,
        "avg_time_sec": avg_time,
        "constraint_satisfaction_rate": constraint_satisfaction_rate,
        "ground_truth_match_rate": ground_truth_match_rate,
        "scores": scores,
        "explanations": explanations
    })
    all_tokens.append(avg_tokens)
    all_times.append(avg_time)

# -------------------------
# Overall summary
# -------------------------
print("\n=== Overall Summary (OptiMUS + Gemini Baseline) ===")
print(f"Mean of per-problem success rates: {np.mean([r['success_rate'] for r in all_results]):.2%}")
print(f"Mean of per-problem avg scores:    {np.mean([r['avg_score'] for r in all_results]):.2f}")
print(f"sd of per-problem avg scores:    {np.std([r['avg_score'] for r in all_results]):.2f}")
print(f"Mean token usage:                   {np.mean(all_tokens):.0f}")
print(f"Mean time per problem (s):         {np.mean(all_times):.2f}")
print(f"Mean constraint satisfaction rate:  {np.mean([r['constraint_satisfaction_rate'] for r in all_results]):.2%}")
print(f"Mean ground truth match rate:       {np.mean([r['ground_truth_match_rate'] for r in all_results]):.2%}")

# Save results
results_path = "optimus_baseline_gem_results.json"
with open(results_path, "w") as f:
    json.dump(all_results, f, indent=4)
print(f"\n✅ Results saved to {results_path}")
