from typing import Dict, Any
from llmtestclient import llm_client

def test_self_consistency(llm_client) -> None:
    """
    Test that the LLM stays consistent within a multi-turn conversation.
    """
    prompt = (
        "User: I have 3 apples."
        "Assistant: Okay."
        "User: I eat one apple. How many apples do I have now?"
    )

    response: Dict[str, Any] = llm_client.generate(prompt)
    answer = response["text"].strip().lower()

    assert "2" in answer, f"Expected '2', got: {answer}"


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
    answer = response["text"].strip().lower()

    assert "paris" in answer, f"Expected 'Paris', got: {answer}"


def test_logical_consistency_facts(llm_client) -> None:
    """
    Test that the LLM follows basic logical deduction in a factual scenario.
    """
    prompt = (
        "Socrates is a man"
        "All men are mortal"
        "Is Socrates mortal? Answer 'Yes' or 'No'."
    )

    response: Dict[str, Any] = llm_client.generate(prompt)
    answer = response["text"].strip().lower()

    assert "yes" in answer, f"Expected 'Yes', got: {answer}"


def test_handles_absurd_question(llm_client) -> None:
    """
    Test that the LLM responds appropriately to a nonsensical question
    without hallucinating a confident false answer.
    """
    prompt = "What color is the number seven's favorite song?"
    response: Dict[str, Any] = llm_client.generate(prompt)
    answer = response["text"].strip().lower()

    # Expected behavior: admit it's nonsensical or not possible
    expected_keywords = ["not sure", "nonsense", "doesn't make sense", "cannot", "unknown", "no favorite"]
    contains_acknowledgment = any(kw in answer for kw in expected_keywords)

    assert contains_acknowledgment, (
        f"Expected the model to acknowledge nonsense, but got:\n{response['text']}"
    )
