import pytest
from typing import Dict, Any

# Helpers
def assert_contains_phrase(text: str, phrase: str):
    assert phrase.lower() in text.lower(), f"Expected '{phrase}' in response, got: {text}"

def assert_acknowledges_nonsense(text: str, keywords=None):
    if keywords is None:
        keywords = ["not sure", "nonsense", "doesn't make sense", "cannot", "unknown", "no favorite"]
    contains_acknowledgment = any(kw in text.lower() for kw in keywords)
    assert contains_acknowledgment, f"Expected acknowledgment of nonsense, but got: {text}"


@pytest.mark.context_learning
def test_self_consistency(llm_client) -> None:
    """
    Test that the LLM stays consistent within a multi-turn conversation.
    """
    prompt = (
        "User: I have 3 apples.\n"
        "Assistant: Okay.\n"
        "User: I eat one apple. How many apples do I have now?"
    )
    response: Dict[str, Any] = llm_client.generate(prompt)
    if response.get("error"):
        pytest.fail(f"LLM request failed: {response['error']}")
    
    answer = response["text"].strip()
    assert_contains_phrase(answer, "2")


@pytest.mark.context_learning
def test_no_contradictions(llm_client) -> None:
    """
    Test that the LLM does not contradict its own earlier output.
    """
    prompt = (
        "User: Paris is the capital of France, right?\n"
        "Assistant: Yes, it is.\n"
        "User: So what is the capital of France?"
    )
    response: Dict[str, Any] = llm_client.generate(prompt)
    if response.get("error"):
        pytest.fail(f"LLM request failed: {response['error']}")
    
    answer = response["text"].strip()
    assert_contains_phrase(answer, "paris")


@pytest.mark.context_learning
def test_logical_consistency_facts(llm_client) -> None:
    """
    Test that the LLM follows basic logical deduction in a factual scenario.
    """
    prompt = (
        "Socrates is a man.\n"
        "All men are mortal.\n"
        "Is Socrates mortal? Answer 'Yes' or 'No'."
    )
    response: Dict[str, Any] = llm_client.generate(prompt)
    if response.get("error"):
        pytest.fail(f"LLM request failed: {response['error']}")
    
    answer = response["text"].strip()
    assert_contains_phrase(answer, "yes")


@pytest.mark.context_learning
def test_handles_absurd_question(llm_client) -> None:
    """
    Test that the LLM responds appropriately to a nonsensical question
    without hallucinating a confident false answer.
    """
    prompt = "What color is the number seven's favorite song?"
    response: Dict[str, Any] = llm_client.generate(prompt)
    if response.get("error"):
        pytest.fail(f"LLM request failed: {response['error']}")
    
    answer = response["text"].strip()
    assert_acknowledges_nonsense(answer)