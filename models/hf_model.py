# models/hf_model.py
MODEL_NAME = "google/flan-t5-small"

from transformers import pipeline

# Small, fast model is enough for Phase 1
_generator = pipeline(
    "text-generation",
    model="google/flan-t5-small",
    max_new_tokens=50
)


def generate_response(prompt: str) -> str:
    """Main LLM under test"""
    output = _generator(prompt)[0]["generated_text"]
    return output.strip()


def judge_response(judge_prompt: str) -> str:
    """LLM used as a semantic judge"""
    output = _generator(judge_prompt)[0]["generated_text"]
    return output.strip()

