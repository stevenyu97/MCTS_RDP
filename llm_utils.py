from curses import raw
import json
import subprocess
import tempfile
import os
import re


#problem_description = ("A wireless communication network includes 4 mobile users, 2 base stations, and 3 frequency carriers.Each mobile user must be assigned to exactly one base station for communication. The system allows each user to transmit on up to 2 different frequency carriers at the same time.The objective of the optimization problem is to minimize the total transmit power used by all users across all carriers. Each user must achieve a minimum signal-to-interference-plus-noise ratio (SINR) of 10 dB on at least one of the carriers they use.A user’s SINR is affected by interference from other users who are transmitting on the same carrier. The power allocated to each user on any given carrier must be between 0 and 1 Watt. Each base station can serve no more than 3 users. The network must determine:Which base station each user connects to,Which carriers each user is assigned to, And how much power each user should transmit with on each carrier, in order to minimize total power while satisfying the connectivity and quality-of-service constraints.")
#problem_description = ("A wireless network consists of 2 mobile users and 1 base station. Each user must connect to the base station and transmit on a single frequency carrier. The goal is to minimize the total transmit power used by both users. Each user must achieve a Signal-to-Interference-plus-Noise Ratio (SINR) of at least 10 dB. SINR is affected by the channel gain between the user and the base station, as well as interference from the other user transmitting on the same frequency. The power allocated to each user must be between 0 and 1 Watt. Your task is to determine the transmit power level for each user such that: 1. Both users meet the SINR requirement 2.The total transmit power is minimized.")


def clean_llm_generated_code(raw_text: str) -> str:
    """
    Cleans LLM-generated code by:
    - Removing markdown-style block markers (```).
    - Stripping off non-code preamble and postscript.
    - Keeping only valid Python code lines (starting with known code keywords).
    """
    code_lines = []
    started = False
    for line in raw_text.splitlines():
        if "```" in line:
            continue
        # Detect start of code
        if not started and re.match(r"^\s*(import|from|def|class|\w+\s*=)", line):
            started = True
        if started:
            # If the line is clearly not code anymore, stop
            if re.match(r"^\s*(In the code|This code|Note that|Explanation|Also,)", line, re.I):
                break
            code_lines.append(line)
    return "\n".join(code_lines)




def generate_structured_sentences(client,description):
    messages = [
        {"role": "system", "content": "You are a systems engineer who breaks optimization problems into structured, descriptive sentences."},
        {"role": "user", "content": f"Break down the following optimization problem into 3 concise descriptive sentences that describe first the context, then the objective, and last the constraints. Each setence must end with a newline character:\n\n{description}"}
    ]
    try:
        response = client.chat.completions.create(
            model="gpt-5",
            messages=messages,
            temperature=1
        )
        token_used = response.usage.total_tokens
        structured_text = response.choices[0].message.content.strip()
        #print(structured_text)
        return [s.strip() for s in structured_text.split("\n") if s.strip()], token_used
    except Exception as e:
        return [f"# Error generating sentences: {str(e)}"], token_used
    
def select_prompt_from_llm(client, sentence: str, depth: int = 0, is_last: bool = False):
    if depth == 0:
        reuse_line = (
            "Choose any suitable optimization library (e.g., PuLP, Pyomo, OR-Tools, CVXPY, mip) and stick with it for subsequent steps.\n"
        )
    else:
        reuse_line = (
            "Reuse the same optimization library already used in prior steps and keep variable names/structure; do not switch libraries.\n"
        )

    system_content = (
        "You are an expert assistant for Python optimization code generation.\n"
        "Transform a single sentence from an optimization problem into a concise, precise prompt "
        "that another LLM can use to generate Python code with an appropriate optimization library.\n"
        + reuse_line +
        "Return only the prompt string (no explanations, code, or reasoning)."
    )

    if is_last:
        user_content = (
            f"The following is a sentence from an optimization problem:\n\n"
            f"\"{sentence}\"\n\n"
            "Write a single, high-quality prompt that instructs an LLM to produce a complete, runnable Python program "
            "that assembles the entire optimization model using the same optimization library as prior steps. "
            "Import required libraries, define all parameters, variables, objective, and constraints, execute the optimization, "
            "and print the final decision variable values and the objective value. "
            "If a single numeric result is requested, also print:\n"
            "  FINAL_ANSWER: <number>\n"
            "Return only the prompt string."
        )
    else:
        user_content = (
            f"The following is a sentence from an optimization problem:\n\n"
            f"\"{sentence}\"\n\n"
            "Write a single, high-quality prompt that instructs an LLM to generate or modify Python code to reflect ONLY this sentence "
            "(e.g., define variables, parameters, a constraint, or an objective) using the same optimization library as prior steps. "
            "Do NOT solve the model or print results at this stage; produce only the incremental code for this sentence. "
            "Return only the prompt string."
        )

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]

    try:
        response = client.chat.completions.create(model="gpt-5", messages=messages)
        token_used = getattr(getattr(response, "usage", None), "total_tokens", 0)
        return response.choices[0].message.content.strip(), token_used
    except Exception as e:
        return f"# Error generating prompt: {str(e)}", 0




def score_prompt(client, prompt):
    messages = [
        {"role": "system", "content": "You score prompts based on descriptiveness and cohesion."},
        {"role": "user", "content": f"Rate this prompt from 1–10 for clarity and detail:\n\n{prompt}\n\nOnly return a number."}
    ]
    
    response = client.chat.completions.create(model="gpt-5", messages=messages)
    token_used = response.usage.total_tokens
    return response.choices[0].message.content.strip(), token_used

def generate_code_from_prompt(client, prompt, previous_code=""):
    if previous_code.strip():
        system_content = (
            "You are a Python optimization expert. Modify the existing optimization code according to the new instruction.\n\n"
            "Requirements:\n"
            "- Preserve the overall structure and variable naming of the existing code.\n"
            "- Make only the necessary edits to incorporate the new instruction.\n"
            "- Use appropriate syntax and functions from the optimization library already used in the code "
            "- Ensure all parameters and variables are correctly defined and assigned before solving.\n"
            "- Ensure the resulting code is complete, runnable, and includes all necessary imports.\n"
            "- Solve the optimization problem and print key variable values and the final objective value.\n"
            "- If a single numeric final quantity is requested, also print:\n"
            "  FINAL_ANSWER: <number>\n"
            "- Do NOT include explanations or comments.\n\n"
            f"Existing Code:\n{previous_code}\n\n"
            f"Instruction:\n{prompt}\n\n"
            "Output only the modified full Python code."
        )
    else:
        system_content = (
            "You are a Python optimization expert. Generate complete, runnable Python code using any suitable optimization library .\n"
            "Requirements:\n"
            "- Import all required libraries and define all variables, parameters, objectives, and constraints.\n"
            "- Ensure the formulation is valid and can be solved by the chosen library.\n"
            "- Assign values to all parameters before solving.\n"
            "- Solve the optimization problem and print key variable values and the final objective value.\n"
            "- If a single numeric final quantity is requested, also print:\n"
            "  FINAL_ANSWER: <number>\n"
            "- Do NOT include explanations or comments.\n\n"
            f"Instruction:\n{prompt}"
        )


    messages = [{"role": "system", "content": system_content}]

    try:
        response = client.chat.completions.create(
            model="gpt-5",
            messages=messages,
            temperature=1
        )
        token_used = response.usage.total_tokens
        code_raw = response.choices[0].message.content.strip()
        return clean_llm_generated_code(code_raw), token_used
    except Exception as e:
        return f"# Error generating code for: {prompt}\n# {str(e)}", 0


def get_final_reward_from_output(code_string, problem_description, client, ground_truth, tolerance):
    """Execute code, capture output, and let LLM assign a reward based on the results."""
    #executed_successfully = False
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
        print(output)

        # if result.returncode == 0:
        #     executed_successfully = True
    except subprocess.TimeoutExpired:
        output = "Execution timed out."
    except Exception as e:
        output = f"Execution failed with error: {str(e)}"
    finally:
        os.remove(tmp_filename)

    # --- 2) Build a strict JSON grading prompt
    gt_block = ""
    if ground_truth is not None and str(ground_truth).strip() != "":
        gt_block = (
            f"\n\nGround-truth numeric answer (if applicable): {ground_truth}\n"
            f"Numeric tolerance for matching (absolute): {tolerance}\n"
            "If a final numeric answer can be extracted from the execution trace, "
            "compute |pred - ground_truth| and set matches_ground_truth = true if the difference "
            f"is <= {tolerance}, else false.\n"
        )
        user_msg = {
        "role": "user",
        "content": (
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
            "Return strictly JSON, e.g.:\n"
            "{\n"
            '  "score": 7.5,\n'
            '  "explanation": "why",\n'
            '  "executed": true,\n'
            '  "constraint_satisfied": false,\n'
            '  "matches_ground_truth": false\n'
            "}\n"
        )
    }
    system_msg = {
        "role": "system",
        "content": (
            "You are a strict grader for optimization code. "
            "Return ONLY a single JSON object with keys: "
            "score (float in [-1,10]), explanation (string), "
            "executed (bool), constraint_satisfied (bool), "
            "matches_ground_truth (bool). Do not add any text outside JSON."
        )
    }
    # Query LLM to evaluate output

    response = client.chat.completions.create(model="gpt-5", messages=[system_msg, user_msg])
    token_used = response.usage.total_tokens
    reply = response.choices[0].message.content.strip()
    print("Grader reply:", reply)

    # --- 4) Parse JSON robustly (fallback to number regex if needed)
    def _safe_json_load(s: str):
        try:
            return json.loads(s)
        except Exception:
            # fallback: extract first {...} block
            m = re.search(r"\{.*\}", s, flags=re.DOTALL)
            if m:
                return json.loads(m.group(0))
            raise

    try:
        graded = _safe_json_load(reply)
        score = float(graded.get("score", -1))
        explanation = str(graded.get("explanation", "")).strip()
        executed_successfully = bool(graded.get("executed", False))
        constraint_satisfied = bool(graded.get("constraint_satisfied", False))
        matches_ground_truth = bool(graded.get("matches_ground_truth", False))

        if executed_successfully and constraint_satisfied and matches_ground_truth:
            score = 10.0

    except Exception:
        # Last-resort heuristic: extract first number; keep your old behavior
        m = re.search(r"-?\d+(\.\d+)?", reply)
        if m:
            score = float(m.group(0))
            explanation = reply
        else:
            raise ValueError(f"Could not parse grader JSON or number from:\n{reply}")

    # Clamp score to [-1,10] just in case
    score = max(-1.0, min(10.0, score))
    return score, explanation, executed_successfully, token_used, constraint_satisfied, matches_ground_truth


def revise_code_based_on_feedback(client, code, feedback, problem_description):
    messages = [
        {"role": "system", "content": "You are a helpful Python assistant that improves optimization code."},
        {"role": "user", "content": (
            f"The following code was generated to solve the optimization problem:\n\n{problem_description}\n\n"
            "Here is the original Python code:\n\n"
            f"{code}\n\n"
            f"However, the code did not perform well. Here's the feedback:\n\n{feedback}\n\n"
            "Please revise and improve the code accordingly, addressing the issues mentioned in the feedback. "
            "Make sure all variables and constraints align with the problem description. Return ONLY the updated Python code, do not provide any explanation back."
        )}
    ]

    response = client.chat.completions.create(model="gpt-5", messages=messages)
    token_used = response.usage.total_tokens
    revised_code = response.choices[0].message.content.strip()
    revised_code = clean_llm_generated_code(revised_code)
    #print(f"\n🛠️ Revised Code:\n{revised_code}")
    return revised_code, token_used