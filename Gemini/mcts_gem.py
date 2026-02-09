import json
import os
import time
import numpy as np
import llm_gemutils
from mcts import SharedNodeMCTS


dataset_path = "/home/ubuntu/LLM_Planning/all_questions.jsonl"
problems = []
with open(dataset_path, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        problems.append({
            "problem_description": obj["en_question"],
            "ground_truth": str(obj["en_answer"])
        })


all_results = []
all_tokens  = []
all_times   = []

results_path = "evaluationMCTS_results_aq_gem.json"

# Ensure file exists before starting
if not os.path.exists(results_path):
    with open(results_path, "w") as f:
        json.dump({"results": []}, f)

for idx, item in enumerate(problems, start=1):
    print(f"\n================ Problem {idx}/{len(problems)} ================")
    problem_description = item["problem_description"]
    ground_truth        = item["ground_truth"]
    tolerance           = 1e-4
    mcts = SharedNodeMCTS()
    results = []
    success_count = 0
    token_counts = []
    run_times = []
    constraint_satisfied_count = 0
    ground_truth_match_count = 0
    per_problem_candidates = []

    for run in range(3):
        try:
            print(f"\n=== Iteration {run + 1} ===")
            start_time = time.time()
            total_tokens = 0

            sentences, token_used = llm_gemutils.generate_structured_sentences(problem_description)
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

                # Generate prompt and score
                is_last = (depth == len(sentences) - 1)

                temp_prompt, token_used = llm_gemutils.select_prompt_from_llm(sentence, depth=depth, is_last=is_last)
                total_tokens += token_used

                temp_score, token_used = llm_gemutils.score_prompt(temp_prompt)
                total_tokens += token_used
                temp_score = float(temp_score)

                # Check for redundancy
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

                # Step 4: Generate code

                code, token_used = llm_gemutils.generate_code_from_prompt(prompt, previous_code=cumulative_code)
                total_tokens += token_used
                cumulative_code = code
                prompt_score_path.append((prompt, score))

            # Step 5: Evaluate reward
            full_code = "# === Auto-generated Optimization Script ===\n\n" + cumulative_code

            reward, explanation, success, token_used, constraint_satisfied, matches_ground_truth = llm_gemutils.get_final_reward_from_output(full_code, problem_description, ground_truth=ground_truth, tolerance=tolerance)
            total_tokens += token_used
            reward_score = float(reward)
            print(explanation, reward_score)

            # === Self-Refine loop ===
            MAX_RETRIES = 2
            REWARD_THRESHOLD = 1
            attempt = 0

            while (not success or reward_score < REWARD_THRESHOLD) and attempt < MAX_RETRIES:
                print(f"\n🔁 Self-Refine Attempt {attempt + 1}")
     
                revised_code, token_used = llm_gemutils.revise_code_based_on_feedback(
                    full_code, explanation, problem_description
                )
                total_tokens += token_used

    
                reward, explanation, success, token_used, constraint_satisfied, matches_ground_truth = llm_gemutils.get_final_reward_from_output(
                    revised_code, problem_description, 
                    ground_truth = ground_truth, tolerance = tolerance
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

            per_problem_candidates.append({
            "run": run + 1,
            "reward": reward_score,
            "success": bool(success),
            "constraint_satisfied": bool(constraint_satisfied),
            "matches_ground_truth": bool(matches_ground_truth),
            "tokens": total_tokens,
            "code": full_code
            })
            # Step 6: MCTS backpropagation
            leaf_node = mcts.build_path(prompt_score_path)
            mcts.backpropagate_reward(leaf_node, reward_score)
            token_counts.append(total_tokens)
            run_times.append(time.time() - start_time)

        except Exception as e:
            print(f"Run {run + 1} failed with error: {e}")
            results.append(-1.0)


    # Choose strongest: highest reward among successful runs; if none succeeded, take highest reward anyway
    successful = [c for c in per_problem_candidates if c["success"]]
    if successful:
        best = max(successful, key=lambda c: c["reward"])
    else:
        best = max(per_problem_candidates, key=lambda c: c["reward"])


    avg_score = float(best["reward"])
    std_score = 0.0
    succ_rate = 1.0 if best["success"] else 0.0
    avg_tokens = float(best["tokens"])
    avg_time = float(np.mean(run_times)) if run_times else 0.0
    constraint_satisfied_rate = 1.0 if best["constraint_satisfied"] else 0.0
    ground_truth_match_rate = 1.0 if best["matches_ground_truth"] else 0.0


    print("\n--- Per-Problem Summary ---")
    print(f"Scores: {results} | mean={avg_score:.2f} std={std_score:.2f}")
    print(f"Success Rate: {succ_rate:.2%}")
    print(f"Avg Token Usage: {avg_tokens:.0f}")
    print(f"Avg Time: {avg_time:.2f} sec")
    print(f"Constraint Satisfaction Rate: {constraint_satisfied_rate:.2%}")
    print(f"Ground Truth Match Rate: {ground_truth_match_rate:.2%}")

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


# Global summary
print("\n=== Overall Summary ===")
print(f"Mean of per-problem success rates: {np.mean([r['success_rate'] for r in all_results]):.2%}")
print(f"Mean of per-problem avg scores:    {np.mean([r['avg_score'] for r in all_results]):.2f}")
print(f"sd of per-problem avg scores:    {np.std([r['avg_score'] for r in all_results]):.2f}")
print(f"Mean token usage:                  {np.mean(all_tokens):.0f}")
print(f"Mean time per problem (s):         {np.mean(all_times):.2f}")
print(f"Mean constraint satisfaction rate:  {np.mean([r['constraint_satisfaction_rate'] for r in all_results]):.2%}")
print(f"Mean ground truth match rate:      {np.mean([r['ground_truth_match_rate'] for r in all_results]):.2%}")