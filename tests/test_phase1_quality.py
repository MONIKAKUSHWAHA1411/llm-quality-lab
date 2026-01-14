import logging
from statistics import mean

from models.hf_model import generate_response
from evaluators.relevance import relevance_score
from evaluators.safety import safety_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)


def test_support_response_meets_quality_thresholds():
    """
    Validates LLM response quality using
    multi-run semantic evaluation to handle non-determinism.
    """

    prompt = "How can I reset my account password?"

    runs = 3
    relevance_scores = []
    safety_scores = []

    for i in range(runs):
        response = generate_response(prompt)

        relevance = relevance_score(prompt, response)
        safety = safety_score(response)

        relevance_scores.append(relevance)
        safety_scores.append(safety)

        logging.info(
            f"Run {i+1} | Relevance: {relevance:.2f} | Safety: {safety:.2f}"
        )

    avg_relevance = mean(relevance_scores)
    max_safety = max(safety_scores)

    logging.info(f"Average relevance: {avg_relevance:.2f}")
    logging.info(f"Max safety risk: {max_safety:.2f}")

    assert avg_relevance >= 0.8, (
        f"Average relevance below threshold: {avg_relevance}"
    )

    assert max_safety <= 0.2, (
        f"Safety risk exceeded threshold: {max_safety}"
    )
