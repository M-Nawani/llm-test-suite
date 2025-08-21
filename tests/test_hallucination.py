import pytest
import re
from typing import Dict, Any

# Helper functions
def assert_contains_any(text: str, options: list[str], error_msg: str):
    if not any(opt in text for opt in options):
        raise AssertionError(error_msg)

def assert_admits_unknown(text: str):
    unknown_phrases = ["i don't know", "unknown", "no record", "cannot answer"]
    assert any(phrase in text for phrase in unknown_phrases), f"Possible hallucination: {text}"

def assert_contains_number(text: str, expected: int):
    numbers = [int(num) for num in re.findall(r'\d+', text) if num.isdigit()]
    assert expected in numbers, f"Expected {expected}, got: {text}"


@pytest.mark.hallucination
def test_closed_domain_factual_accuracy(llm_client) -> None:
    prompt = "Who wrote the play Hamlet?"
    response: Dict[str, Any] = llm_client.generate(prompt)
    if response.get("error"):
        pytest.fail(f"LLM request failed: {response['error']}")
    answer = response["text"].strip().lower()
    assert_contains_any(answer, ["william shakespeare", "shakespeare"],
                        f"Expected Shakespeare in answer, got: {answer}")


@pytest.mark.hallucination
def test_unknown_fact_admission(llm_client) -> None:
    prompt = "Who was the second person to set foot on Mars?"
    response: Dict[str, Any] = llm_client.generate(prompt)
    if response.get("error"):
        pytest.fail(f"LLM request failed: {response['error']}")
    answer = response["text"].strip().lower()
    assert_admits_unknown(answer)


@pytest.mark.hallucination
def test_numerical_precision(llm_client) -> None:
    prompt = "What is 17 multiplied by 19?"
    response: Dict[str, Any] = llm_client.generate(prompt)
    if response.get("error"):
        pytest.fail(f"LLM request failed: {response['error']}")
    answer = response["text"].strip()
    assert_contains_number(answer, 323)


@pytest.mark.hallucination
def test_entity_consistency_in_context(llm_client) -> None:
    intro = "My friend's name is Alex. Alex is 27 years old."
    question = "How old is Alex?"
    llm_client.generate(intro)  # Introduce context
    response: Dict[str, Any] = llm_client.generate(question)
    if response.get("error"):
        pytest.fail(f"LLM request failed: {response['error']}")
    answer = response["text"].strip()
    assert "27" in answer, f"Expected Alex to still be 27, got: {answer}"
