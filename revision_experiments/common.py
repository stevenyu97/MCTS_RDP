"""Shared helpers for this-round revision experiments (Tables 4 and 5)."""
from __future__ import annotations

import json
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import llm_utils
from mcts import SharedNodeMCTS

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = ROOT / "all_questions.jsonl"
DEFAULT_EVAL_JSON = ROOT / "evaluationMCTS_results.json"
REVISION_DIR = ROOT / "revision_experiments" / "traces"

MODEL = "gpt-5"
TEMPERATURE = 1
TOLERANCE = 1e-4
TIMEOUT_SEC = 30
SELF_REFINE_MAX_RETRIES = 2
REWARD_THRESHOLD = 1.0
MCTS_SIMULATIONS = 3
# Search-then-apply: exactly 10% search slots per block (3 of 30).
BLOCK_SIZE = 30
SEARCH_FRACTION = 0.10
A_SEARCH_COUNT = max(1, round(BLOCK_SIZE * SEARCH_FRACTION))  # 3 => 10%


def configure_stpa_protocol(block_size: int = 30, search_fraction: float = 0.10) -> None:
    """Set block-wise search-then-apply parameters (default 10% search)."""
    global BLOCK_SIZE, SEARCH_FRACTION, A_SEARCH_COUNT
    BLOCK_SIZE = block_size
    SEARCH_FRACTION = search_fraction
    A_SEARCH_COUNT = max(1, round(BLOCK_SIZE * SEARCH_FRACTION))


def load_problems(path: Path = DEFAULT_DATASET) -> List[Dict[str, str]]:
    problems = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            problems.append({
                "problem_description": obj["en_question"],
                "ground_truth": str(obj.get("en_answer", "")).strip(),
            })
    return problems


def load_eval_summaries(path: Path = DEFAULT_EVAL_JSON) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def historical_stratum(row: Dict[str, Any]) -> str:
    """Stratify by archived MCTS summary labels."""
    if row.get("success_rate", 0) < 1.0 or row.get("avg_score", 0) < 0:
        return "failed"
    if row.get("ground_truth_match_rate", 0) >= 1.0:
        return "optimal"
    return "suboptimal"


def actual_stratum(det: Dict[str, Any]) -> str:
    if not det.get("executed"):
        return "failed"
    if det.get("matches_ground_truth"):
        return "optimal"
    return "suboptimal"


def pick_stratified_indices(
    eval_rows: List[Dict[str, Any]],
    per_stratum: int = 15,
    seed: int = 42,
) -> List[int]:
    rng = random.Random(seed)
    buckets: Dict[str, List[int]] = {"optimal": [], "suboptimal": [], "failed": []}
    for row in eval_rows:
        buckets[historical_stratum(row)].append(int(row["problem_index"]))

    chosen: List[int] = []
    for key in ("optimal", "suboptimal", "failed"):
        pool = buckets[key][:]
        rng.shuffle(pool)
        chosen.extend(pool[:per_stratum])
    return sorted(set(chosen))


def _parse_numeric_output(output: str) -> Optional[float]:
    if not output:
        return None
    m = re.search(
        r"FINAL_ANSWER\s*[:=]\s*(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)",
        output,
        re.IGNORECASE,
    )
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    for pat in (
        r"(?:objective|optimal value|objective value|profit|cost|total)\s*[:=]\s*(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)",
        r"(?:Optimal|Objective)\s*\(?\s*value\s*\)?\s*[:=]\s*(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)",
    ):
        matches = re.findall(pat, output, re.IGNORECASE)
        if matches:
            try:
                return float(matches[-1])
            except ValueError:
                continue
    return None


def deterministic_checks(
    execution_trace: str,
    returncode: int,
    ground_truth: str,
    tolerance: float = TOLERANCE,
) -> Dict[str, Any]:
    """Post-hoc return-code and numeric-GT checks used by the Table 4 reliability study."""
    timed_out = "Execution timed out" in execution_trace
    executed = returncode == 0 and not timed_out
    pred = _parse_numeric_output(execution_trace)
    try:
        gt = float(str(ground_truth).strip())
    except (TypeError, ValueError):
        gt = None
    matches = pred is not None and gt is not None and abs(pred - gt) <= tolerance
    return {
        "executed": executed,
        "matches_ground_truth": matches,
        "constraint_satisfied": executed and pred is not None,
        "pred": pred,
        "returncode": returncode,
        "timed_out": timed_out,
    }


def generate_oneshot_code(client, problem_description: str) -> Tuple[str, int]:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a Python optimization expert. Solve the following optimization problem using any suitable Python library.\n"
                "Generate complete, runnable code that defines variables, constraints, and objective, executes the optimization, and prints the final result.\n"
                "If a single numeric value is requested, also print:\n"
                "  FINAL_ANSWER: <number>\n"
                "Do not include explanations or comments."
            ),
        },
        {
            "role": "user",
            "content": f"Problem: {problem_description}\n\nGenerate the complete Python code now.",
        },
    ]
    resp = client.chat.completions.create(
        model=MODEL, messages=messages, temperature=TEMPERATURE
    )
    code = llm_utils.clean_llm_generated_code(resp.choices[0].message.content.strip())
    return code, resp.usage.total_tokens


def grade_code(
    client,
    code: str,
    problem_description: str,
    ground_truth: str,
    tolerance: float = TOLERANCE,
) -> Dict[str, Any]:
    score, explanation, executed, tokens, constraint_ok, matches_gt = (
        llm_utils.get_final_reward_from_output(
            code, problem_description, client, ground_truth, tolerance
        )
    )
    return {
        "code": code,
        "score": float(score),
        "explanation": explanation,
        "executed": bool(executed),
        "constraint_satisfied": bool(constraint_ok),
        "matches_ground_truth": bool(matches_gt),
        "tokens": int(tokens),
    }


def run_cot_once(client, problem_description: str) -> Tuple[str, int]:
    total_tokens = 0
    sentences, tok = llm_utils.generate_structured_sentences(client, problem_description)
    total_tokens += tok

    cumulative_code = ""
    for i, sentence in enumerate(sentences):
        is_last = i == len(sentences) - 1
        prompt, tok = llm_utils.select_prompt_from_llm(client, sentence, i, is_last)
        total_tokens += tok
        code, tok = llm_utils.generate_code_from_prompt(
            client, prompt, previous_code=cumulative_code
        )
        total_tokens += tok
        cumulative_code = code

    full_code = "# === CoT ===\n\n" + cumulative_code
    return full_code, total_tokens


def run_self_refine_once(
    client,
    problem_description: str,
    ground_truth: str,
    tolerance: float = TOLERANCE,
) -> Tuple[str, int, Dict[str, Any]]:
    code, gen_tokens = generate_oneshot_code(client, problem_description)
    graded = grade_code(client, code, problem_description, ground_truth, tolerance)
    total_tokens = gen_tokens + graded["tokens"]

    refines = 0
    while refines < SELF_REFINE_MAX_RETRIES and (
        not graded["executed"] or graded["score"] < REWARD_THRESHOLD
    ):
        code, fb_tokens = llm_utils.revise_code_based_on_feedback(
            client, code, graded["explanation"], problem_description
        )
        total_tokens += fb_tokens
        graded = grade_code(client, code, problem_description, ground_truth, tolerance)
        total_tokens += graded["tokens"]
        refines += 1

    graded["tokens"] = total_tokens
    graded["code"] = code
    return code, total_tokens, graded


def _compact_prompt_path(prompt_score_path, keep=3) -> str:
    trimmed = prompt_score_path[:keep]
    lines = []
    for i, (p, s) in enumerate(trimmed, 1):
        lines.append(f"[Step {i}] score={s:.2f}\n{p.strip()}")
    return "\n".join(lines)


def build_examples_only_prompt(problem_description: str, exemplar: str) -> str:
    head = (
        "You are a Python optimization expert.\n"
        "Use the compact exemplar below to structure the solution. Keep code concise and runnable.\n\n"
        "=== EXEMPLAR (best-in-block) ===\n"
    )
    ex_block = exemplar if exemplar else "[No exemplar available]\n"
    tail = (
        "\n=== TARGET PROBLEM ===\n"
        f"{problem_description.strip()}\n\n"
        "Produce a complete, runnable solution. Print key results and FINAL_ANSWER if applicable."
    )
    return head + ex_block + tail


class BlockExemplarCache:
    def __init__(self) -> None:
        self._block_best: Dict[int, Dict[str, Any]] = {}

    @staticmethod
    def _should_replace(existing: Optional[Dict[str, Any]], reward: float, tokens: int) -> bool:
        if existing is None:
            return True
        if reward > existing["reward"]:
            return True
        if reward < existing["reward"]:
            return False
        return tokens < existing["tokens"]

    def update_from_search_runs(self, block_id: int, candidates: List[Dict[str, Any]]) -> None:
        if not candidates:
            return
        successful = [c for c in candidates if c.get("executed")]
        pool = successful if successful else candidates
        best = max(pool, key=lambda c: (c["score"], -c.get("tokens", 0)))
        path = best.get("path") or []
        exemplar = _compact_prompt_path(path, keep=3)
        prev = self._block_best.get(block_id)
        if self._should_replace(prev, float(best["score"]), int(best.get("tokens", 0))):
            self._block_best[block_id] = {
                "reward": float(best["score"]),
                "tokens": int(best.get("tokens", 0)),
                "example": exemplar,
            }

    def get_exemplar(self, block_id: int) -> str:
        entry = self._block_best.get(block_id)
        return entry["example"] if entry else ""


def paper_protocol_condition(dataset_problem_index: int) -> str:
    within_blk = (dataset_problem_index - 1) % BLOCK_SIZE
    return "SEARCH" if within_blk < A_SEARCH_COUNT else "APPLY"


def run_mcts_apply_once(
    client,
    problem_description: str,
    ground_truth: str,
    exemplar: str,
    tolerance: float = TOLERANCE,
) -> Dict[str, Any]:
    composed = build_examples_only_prompt(problem_description, exemplar)
    run_start = time.time()
    run_tokens = 0

    code, tok = llm_utils.generate_code_from_prompt(client, composed, previous_code="")
    run_tokens += tok
    full_code = code
    graded = grade_code(client, full_code, problem_description, ground_truth, tolerance)
    run_tokens += graded["tokens"]

    attempt = 0
    while (
        (not graded["executed"] or graded["score"] < REWARD_THRESHOLD)
        and attempt < SELF_REFINE_MAX_RETRIES
    ):
        code, fb_tokens = llm_utils.revise_code_based_on_feedback(
            client, full_code, graded["explanation"], problem_description
        )
        run_tokens += fb_tokens
        graded = grade_code(client, code, problem_description, ground_truth, tolerance)
        run_tokens += graded["tokens"]
        full_code = code
        attempt += 1

    graded["code"] = full_code
    graded["tokens"] = run_tokens
    graded["time"] = time.time() - run_start
    graded["condition"] = "APPLY"
    return graded


def run_mcts_search_then_apply(
    client,
    dataset_problem_index: int,
    problem_description: str,
    ground_truth: str,
    num_candidates: int,
    exemplar_cache: BlockExemplarCache,
    tolerance: float = TOLERANCE,
) -> Dict[str, Any]:
    block_id = (dataset_problem_index - 1) // BLOCK_SIZE
    condition = paper_protocol_condition(dataset_problem_index)

    if condition == "SEARCH":
        candidates, _ = run_mcts_rollouts_once(
            client, problem_description, ground_truth, num_rollouts=num_candidates, tolerance=tolerance
        )
        for c in candidates:
            c["condition"] = "SEARCH"
        exemplar_cache.update_from_search_runs(block_id, candidates)
        agg = aggregate_candidate_runs(candidates)
        agg["mcts_condition"] = "SEARCH"
        agg["block_id"] = block_id
        return agg

    exemplar = exemplar_cache.get_exemplar(block_id)
    runs = []
    for cand in range(num_candidates):
        graded = run_mcts_apply_once(
            client, problem_description, ground_truth, exemplar, tolerance
        )
        graded["candidate"] = cand + 1
        runs.append(graded)

    agg = aggregate_candidate_runs(runs)
    agg["mcts_condition"] = "APPLY"
    agg["block_id"] = block_id
    agg["had_exemplar"] = bool(exemplar)
    return agg


def sample_dataset_indices(num: int, dataset_size: int, seed: int = 42) -> List[int]:
    if num > dataset_size:
        raise ValueError(f"Cannot sample {num} problems from dataset of size {dataset_size}")
    rng = random.Random(seed)
    return sorted(rng.sample(range(1, dataset_size + 1), num))


def run_mcts_rollouts_once(
    client,
    problem_description: str,
    ground_truth: str,
    num_rollouts: int = MCTS_SIMULATIONS,
    tolerance: float = TOLERANCE,
) -> Tuple[List[Dict[str, Any]], int]:
    """Run MCTS search rollouts (shared tree) as in SEARCH-mode mcts_oneshot."""
    mcts = SharedNodeMCTS()
    candidates: List[Dict[str, Any]] = []
    total_tokens = 0

    for run in range(num_rollouts):
        run_start = time.time()
        run_tokens = 0

        sentences, tok = llm_utils.generate_structured_sentences(client, problem_description)
        run_tokens += tok

        cumulative_code = ""
        prompt_score_path = []
        parent_at_depth = {0: mcts.root}

        for depth, sentence in enumerate(sentences):
            parent_node = parent_at_depth[depth]
            if parent_node.children:
                current_node = mcts.select_node_by_uct(parent_node)
            else:
                current_node = parent_node

            is_last = depth == len(sentences) - 1
            temp_prompt, tok = llm_utils.select_prompt_from_llm(
                client, sentence, depth=depth, is_last=is_last
            )
            run_tokens += tok
            temp_score, tok = llm_utils.score_prompt(client, temp_prompt)
            run_tokens += tok
            temp_score = float(temp_score)

            found_sibling = None
            for sibling in parent_node.children:
                if sibling.score == temp_score:
                    found_sibling = sibling
                    break

            if found_sibling:
                current_node = found_sibling
            else:
                current_node = mcts.expand_node(parent_node, temp_prompt, temp_score)

            parent_at_depth[depth + 1] = current_node
            code, tok = llm_utils.generate_code_from_prompt(
                client, current_node.prompt, previous_code=cumulative_code
            )
            run_tokens += tok
            cumulative_code = code
            prompt_score_path.append((current_node.prompt, current_node.score))

        full_code = "# === MCTS-RDP ===\n\n" + cumulative_code
        graded = grade_code(
            client, full_code, problem_description, ground_truth, tolerance
        )
        run_tokens += graded["tokens"]

        attempt = 0
        while (
            (not graded["executed"] or graded["score"] < REWARD_THRESHOLD)
            and attempt < SELF_REFINE_MAX_RETRIES
        ):
            code, fb_tokens = llm_utils.revise_code_based_on_feedback(
                client, full_code, graded["explanation"], problem_description
            )
            run_tokens += fb_tokens
            graded = grade_code(client, code, problem_description, ground_truth, tolerance)
            run_tokens += graded["tokens"]
            full_code = code
            attempt += 1

        graded["code"] = full_code
        graded["tokens"] = run_tokens
        graded["run"] = run + 1
        graded["time"] = time.time() - run_start
        graded["path"] = prompt_score_path
        graded["condition"] = "SEARCH"
        candidates.append(graded)
        total_tokens += run_tokens

        leaf_node = mcts.build_path(prompt_score_path)
        mcts.backpropagate_reward(leaf_node, graded["score"])

    return candidates, total_tokens


def aggregate_candidate_runs(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    scores = [r["score"] for r in runs]
    best_idx = max(range(len(runs)), key=lambda i: runs[i]["score"])
    best = runs[best_idx]
    return {
        "runs": runs,
        "mean_score": float(sum(scores) / len(scores)),
        "best_score": float(best["score"]),
        "mean_executed": float(sum(r["executed"] for r in runs) / len(runs)),
        "best_executed": bool(best["executed"]),
        "mean_constraint_satisfied": float(
            sum(r["constraint_satisfied"] for r in runs) / len(runs)
        ),
        "best_constraint_satisfied": bool(best["constraint_satisfied"]),
        "mean_matches_ground_truth": float(
            sum(r["matches_ground_truth"] for r in runs) / len(runs)
        ),
        "best_matches_ground_truth": bool(best["matches_ground_truth"]),
        "total_tokens": int(sum(r.get("tokens", 0) for r in runs)),
    }
