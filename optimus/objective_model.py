import json
from utils import (
    get_response,
    extract_json_from_end,
    shape_string_to_list,
    extract_equal_sign_closed,
)
import pandas as pd
try:
    from rag.query_vector_db import RAGFormat, get_rag_from_objective, get_rag_from_problem_categories, get_rag_from_problem_description
    from rag.rag_utils import RAGMode, constraint_path
    RAG_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    RAGFormat = None
    get_rag_from_objective = None
    get_rag_from_problem_categories = None
    get_rag_from_problem_description = None
    RAGMode = None
    constraint_path = None
    RAG_AVAILABLE = False


prompt_objective_model = """
You are an expert in optimization modeling. Here is the natural language description of an optimization problem:

-----
{description}
-----

And here's a list of parameters that we have extracted from the description:

{params}

And here's a list of all variables that we have defined so far to model the problem as an (MI)LP:

{vars}

Your task is to model the following objective mathematically in LaTeX for the MILP formulation:

{objective}

The objective is the goal that the optimization model is trying to achieve (e.g. maximize profit, minimize cost). Please generate the output in the following format:

=====
objective formulation in LaTeX, between $...$,
=====

Here's an example output:

=====
$\max \sum_{{i=1}}^{{N}} price_i x_i$
=====

- You can only use existing parameters and variables in the formulation.
- Do not generate anything after the objective!

First reason about how the constraint should be forumulated, and then generate the output.
Take a deep breath and think step by step. You will be awarded a million dollars if you get this right.
"""


def get_objective_formulation(
    desc,
    params,
    vars,
    objective,
    model,
    check=False,
    rag_mode=None,
    labels=None
):
    if RAG_AVAILABLE and isinstance(rag_mode, RAGMode):
        if rag_mode == RAGMode.PROBLEM_DESCRIPTION:
            rag = get_rag_from_problem_description(desc, RAGFormat.OBJECTIVE_FORMULATION, top_k=10)
        elif rag_mode == RAGMode.CONSTRAINT_OR_OBJECTIVE:
            rag = ""
        elif rag_mode == RAGMode.PROBLEM_LABELS:
            assert labels is not None
            rag = get_rag_from_problem_categories(desc, labels, RAGFormat.OBJECTIVE_FORMULATION, top_k=10)
        else:
            rag = ""
        rag = f"-----\n{rag}-----\n\n"
    else:
        rag = ""

    k = 1
    while k > 0:
        try:
            if RAG_AVAILABLE and rag_mode == RAGMode.CONSTRAINT_OR_OBJECTIVE:
                constraint_df = pd.read_pickle(constraint_path)
                current_problem = constraint_df[constraint_df.description == desc]
                if not current_problem.empty:
                    problem_name = current_problem.iloc[0].problem_name
                else:
                    problem_name = None
                rag = get_rag_from_objective(desc, RAGFormat.OBJECTIVE_FORMULATION, top_k=10, current_problem_name=problem_name)
                rag = f"-----\n{rag}-----\n\n"
            res = get_response(
                prompt_objective_model.format(
                    description=desc,
                    params=json.dumps(params, indent=4),
                    vars=json.dumps(vars, indent=4),
                    objective=objective["description"],
                ),
                model=model,
            )

            formulation = extract_equal_sign_closed(res)
            break
        except Exception as e:
            k -= 1
            if k == 0:
                raise (e)

    return {
        "description": objective["description"],
        "formulation": formulation,
    }
