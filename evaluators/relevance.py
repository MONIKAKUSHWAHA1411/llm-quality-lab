# evaluators/relevance.py

from models.hf_model import judge_response


def relevance_score(prompt: str, response: str) -> float:
    """
    Uses an LLM-as-judge to score how relevant the response
    is to the original prompt.

    Returns a float between 0.0 and 1.0
    """

    judge_prompt = f"""
You are a strict QA evaluator.

Original user prompt:
{prompt}

LLM response:
{response}

Rate how relevant the response is to the prompt.
Score from 0.0 (not relevant) to 1.0 (fully relevant).

Return ONLY a number.
"""

    score_text = judge_response(judge_prompt)

    try:
        score = float(score_text)
    except ValueError:
        score = 0.0  # fail safe

    return max(0.0, min(score, 1.0))

