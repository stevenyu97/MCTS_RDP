import json
import subprocess
import tempfile
import os
import re
import google.generativeai as genai

# --- CONFIGURATION ---

genai.configure(api_key="")

# Default model for the helper (grader/structurer)
MODEL_NAME = "gemini-2.5-pro"

def clean_llm_generated_code(raw_text: str) -> str:
    """
    Cleans LLM-generated code by removing markdown markers and non-code text.
    """
    code_lines = []
    started = False
    # Clean up markdown immediately
    raw_text = raw_text.replace("```python", "").replace("```", "")
    
    for line in raw_text.splitlines():
        # Detect start of code
        if not started and re.match(r"^\s*(import|from|def|class|\w+\s*=)", line):
            started = True
        if started:
            # If the line is clearly not code anymore, stop
            if re.match(r"^\s*(In the code|This code|Note that|Explanation|Also,)", line, re.I):
                break
            code_lines.append(line)
    return "\n".join(code_lines)

def call_gemini(system_instruction, user_prompt, json_mode=False, temperature=1.0):
    """
    Helper function to initialize the model with a specific system prompt 
    and generate content using genai.types for safety settings.
    """
    try:
        # Configure generation settings
        generation_config = {
            "temperature": temperature,
        }
        
        if json_mode:
            generation_config["response_mime_type"] = "application/json"

        # Initialize model
        model = genai.GenerativeModel(
            model_name=MODEL_NAME,
            system_instruction=system_instruction,
            generation_config=generation_config
        )
        
        safety_settings = {
            genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
            genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
            genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: genai.types.HarmBlockThreshold.BLOCK_NONE,
            genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: genai.types.HarmBlockThreshold.BLOCK_NONE,
        }

        response = model.generate_content(user_prompt, safety_settings=safety_settings)
        

        token_count = response.usage_metadata.total_token_count
        return response.text.strip(), token_count

    except Exception as e:
        print(f"Gemini API Error: {e}")
        return str(e), 0

def generate_structured_sentences(description):
    system_msg = (
        "You are an expert in mathematical optimization. "
        "You must decompose problems without losing any numeric information."
    )

    user_msg = f"""
Decompose the optimization problem into **EXACTLY 3 sentences**, each on its own line:

1) Context & decision variables:
   - Mention all decision variables/items/resources.
   - Include EVERY numeric value tied to context (budgets, supplies, set sizes, times, etc.) with units.
2) Objective:
   - State minimize/maximize.
   - Include the exact objective expression in words using the given numeric coefficients.
3) Constraints:
   - Describe ALL constraints.
   - Explicitly include every numeric bound/ratio/percentage and units.
   - If there are multiple constraints, list them in the same sentence separated by semicolons.

Rules:
- Do NOT invent new variables, resources, or numbers.
- Do NOT change any number; copy them exactly as written.
- Keep units (minutes, dollars, trips, %, etc.).
- If a percent appears, keep it as a percent (e.g., 40%) and also include decimal form if it’s natural (40% = 0.40).
- Return ONLY the 3 sentences, no bullets, no extra text.

PROBLEM:
{description}
"""

    content, token_used = call_gemini(system_msg, user_msg)

    if "Error" in content:
        return [f"# Error generating sentences"], token_used

    return [s.strip() for s in content.split("\n") if s.strip()], token_used


def select_prompt_from_llm(sentence: str, depth: int = 0, is_last: bool = False):
    # Library guidance: allow any library, but enforce consistency across steps
    if depth == 0:
        reuse_line = (
            "Choose ONE suitable Python optimization library (e.g., PuLP, OR-Tools, CVXPY, SciPy, mip, Pyomo, gurobipy) "
            "and explicitly state which you choose. You MUST stick with that same library for ALL subsequent steps.\n"
        )
    else:
        reuse_line = (
            "Reuse the SAME optimization library, model object, and variable names already used in prior steps. "
            "Do NOT switch libraries or rename variables.\n"
        )

    system_content = (
        "You are an expert assistant for optimization code prompting.\n"
        "Transform ONE decomposed sentence into a concise prompt that another LLM will use to PATCH existing Python optimization code.\n"
        + reuse_line +
        "Prompt-writing rules you MUST follow:\n"
        "- The prompt must instruct PATCHING existing code, NOT rewriting from scratch.\n"
        "- Do NOT allow inventing new decision variables, constraints, objectives, or numbers.\n"
        "- Only use numeric values explicitly present in the sentence or already present in prior code.\n"
        "- Preserve objective sense (min/max) and coefficients unless this sentence is explicitly the objective sentence.\n"
        "- Preserve variable names/meaning unless this sentence explicitly introduces a new variable.\n"
        "Return ONLY the prompt string (no explanations, no code).\n"
    )

    if is_last:

        user_content = (
            f"This is the FINAL sentence from an optimization problem:\n\n"
            f"\"{sentence}\"\n\n"
            "Write ONE high-quality prompt telling an LLM to PATCH the existing code to incorporate ONLY what this sentence adds.\n"
            "Your prompt MUST include these strict patch rules:\n"
            "- Do NOT restart or rewrite the program from scratch.\n"
            "- Do NOT create a new model if one already exists; modify the existing one.\n"
            "- Do NOT rename variables or introduce new ones unless explicitly required by this sentence.\n"
            "- Do NOT change any existing numeric coefficients; only add missing ones from this sentence.\n"
            "- If this sentence describes constraints, ONLY add those constraints.\n"
            "- Keep the same objective already defined unless this sentence explicitly defines/redefines the objective.\n"
            "- After patching, ensure the final code solves the model and prints variable values and the objective value.\n"
            "- Print: FINAL_ANSWER: <objective value>\n"
            "Return ONLY the prompt string."
        )
    else:

        user_content = (
            f"This is ONE sentence from an optimization problem:\n\n"
            f"\"{sentence}\"\n\n"
            "Write ONE high-quality prompt telling an LLM to PATCH the existing code to reflect ONLY this sentence.\n"
            "Your prompt MUST include:\n"
            "- Patch existing code; do NOT rewrite from scratch.\n"
            "- Reuse the same library/model/variable names from prior steps.\n"
            "- Do NOT invent new numbers/variables/constraints beyond what the sentence states.\n"
            "- If this sentence introduces variables/parameters, add ONLY those definitions.\n"
            "- If this sentence defines the objective, set/replace ONLY the objective.\n"
            "- If this sentence defines constraints, add ONLY those constraint lines.\n"
            "- Do NOT solve or print results yet.\n"
            "Return ONLY the prompt string."
        )

    return call_gemini(system_content, user_content)  



def score_prompt(prompt):
    system_msg = "You score prompts based on descriptiveness and cohesion."
    user_msg = f"Rate this prompt from 1–10 for clarity and detail:\n\n{prompt}\n\nOnly return a number."
    return call_gemini(system_msg, user_msg)

def generate_code_from_prompt(prompt, previous_code=""):

    if previous_code.strip():
        system_content = (
            "You are a Python optimization expert. "
            "You MUST patch the existing code instead of rewriting.\n\n"
            "Patch rules:\n"
            "- Preserve the model, variable names, and library already used.\n"
            "- Do NOT introduce new decision variables or remove existing ones unless explicitly required.\n"
            "- Do NOT introduce any numeric values that are not in the instruction or existing code.\n"
            "- Do NOT rename variables.\n"
            "- Do NOT create a new problem template or restart from scratch.\n"
            "- Only add/modify the objective or constraints needed by the instruction.\n"
            "- Keep the code runnable and complete.\n"
            "- Print FINAL_ANSWER: <number> as the objective value.\n"
            "- Output ONLY full modified Python code, no explanations.\n"
        )

        user_prompt = (
            f"EXISTING CODE:\n{previous_code}\n\n"
            f"NEW INSTRUCTION:\n{prompt}\n\n"
            "Return the full patched code."
        )

    else:
        system_content = (
            "You are a Python optimization expert. Generate a complete optimization script.\n\n"
            "Generation rules:\n"
            "- Use ONLY numbers/variables stated in the instruction.\n"
            "- Do NOT invent extra decision variables, constraints, or data.\n"
            "- Define variables, objective, constraints, solve, print variables and FINAL_ANSWER.\n"
            "- Output ONLY runnable Python code, no explanations.\n"
        )

        user_prompt = (
            f"INSTRUCTION:\n{prompt}\n\n"
            "Return full optimization code."
        )
    code_raw, token_used = call_gemini(system_content, user_prompt, temperature=1.0) 
    return clean_llm_generated_code(code_raw), token_used

def get_final_reward_from_output(code_string, problem_description, ground_truth, tolerance):
    """Execute code, capture output, and let LLM assign a reward based on the results."""
    
    # --- 1) Execution Phase (Standard Python Subprocess) ---
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.py', delete=False) as tmp:
        tmp.write(code_string)
        tmp_filename = tmp.name

    try:
        result = subprocess.run(
            ["python3", tmp_filename],
            capture_output=True,
            text=True,
            timeout=30
        )
        output = result.stdout + "\n" + result.stderr
        # print(f"--- Execution Output ---\n{output}")
    except subprocess.TimeoutExpired:
        output = "Execution timed out."
    except Exception as e:
        output = f"Execution failed with error: {str(e)}"
    finally:
        os.remove(tmp_filename)

    # --- 2) Grading Phase (Gemini with JSON Mode) ---
    gt_block = ""
    if ground_truth is not None and str(ground_truth).strip() != "":
        gt_block = (
            f"\n\nGround-truth numeric answer (if applicable): {ground_truth}\n"
            f"Numeric tolerance for matching (absolute): {tolerance}\n"
            "If a final numeric answer can be extracted from the execution trace, "
            "compute |pred - ground_truth| and set matches_ground_truth = true if the difference "
            f"is <= {tolerance}, else false.\n"
        )

    user_msg = (
        "Problem description:\n"
        f"{problem_description}\n\n"
        "Execution trace (stdout + stderr):\n"
        f"{output}\n"
        f"{gt_block}\n"
        "Scoring rubric:\n"
        "- If the program did not execute (timeout, runtime error, syntax error), set score = -1 and executed=false.\n"
        "- Otherwise, start from a base score and increase for:\n"
        "  (i) Executability and completeness of results, \n"
        "  (ii) Constraint satisfaction (feasibility), \n"
        "  (iii) Objective quality and plausibility, \n"
        "  (iv) (Optional) closeness to ground-truth if provided.\n"
        "Notes:\n"
        "- Keep the final score in [-1, 10].\n"
        "- If no meaningful result can be extracted despite successful execution, you may assign a low nonnegative score.\n"
    )

    system_msg = (
        "You are a strict grader for optimization code. "
        "Return ONLY a single JSON object with keys: "
        "score (float in [-1,10]), explanation (string), "
        "executed (bool), constraint_satisfied (bool), "
        "matches_ground_truth (bool)."
    )

    # Call Gemini with JSON mode enabled
    reply, token_used = call_gemini(system_msg, user_msg, json_mode=True)
    # print("Grader reply:", reply)

    # --- 3) Parse JSON ---
    try:
        graded = json.loads(reply)
        score = float(graded.get("score", -1))
        explanation = str(graded.get("explanation", "")).strip()
        executed_successfully = bool(graded.get("executed", False))
        constraint_satisfied = bool(graded.get("constraint_satisfied", False))
        matches_ground_truth = bool(graded.get("matches_ground_truth", False))

        # Force perfect score when constraints are met and ground truth matches
        if executed_successfully and constraint_satisfied and matches_ground_truth:
            score = 10.0

    except Exception as e:
        print(f"JSON Parsing failed: {e}")
        score = -1
        explanation = "Failed to parse grader output."
        executed_successfully = False
        constraint_satisfied = False
        matches_ground_truth = False

    # Clamp score
    score = max(-1.0, min(10.0, score))
    
    return score, explanation, executed_successfully, token_used, constraint_satisfied, matches_ground_truth


def revise_code_based_on_feedback(code, feedback, problem_description):
    system_msg = "You are a helpful Python assistant that improves optimization code."
    user_msg = (
        f"The following code was generated to solve the optimization problem:\n\n{problem_description}\n\n"
        "Here is the original Python code:\n\n"
        f"{code}\n\n"
        f"However, the code did not perform well. Here's the feedback:\n\n{feedback}\n\n"
        "Please revise and improve the code accordingly, addressing the issues mentioned in the feedback. "
        "Make sure all variables and constraints align with the problem description. Return ONLY the updated Python code, do not provide any explanation back."
    )

    revised_code_raw, token_used = call_gemini(system_msg, user_msg)
    revised_code = clean_llm_generated_code(revised_code_raw)
    return revised_code, token_used