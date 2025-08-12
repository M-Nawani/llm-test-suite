from typing import Dict, Any
from llmtestclient import llm_client


def test_generate_non_empty_response(llm_client) -> None:
    """
    Test that the LLM generates a non-empty, sufficiently long response
    to a simple prompt.
    """
    response: Dict[str, Any] = llm_client.generate("How are you?")
    text: str = response.get("text", "").strip()

    assert text, "Response text is empty."
    assert len(text) > 10, f"Response too short: {len(text)} characters."
    print(f"Response length: {len(text)} characters.")


def test_follow_simple_instruction(llm_client) -> None:
    """
    Test that the LLM follows a simple instruction by naming at least
    one color from a predefined list.
    """
    response: Dict[str, Any] = llm_client.generate(
        "Name three colors. Just list the colors."
    )
    text_lower: str = response.get("text", "").lower()

    colors = {
        "violet", "indigo", "blue", "green", "yellow",
        "orange", "red", "white", "black", "grey"
    }
    found_colors = {color for color in colors if color in text_lower}

    assert found_colors, f"No expected colors found in: {response.get('text', '')}"


def test_answer_simple_fact_question(llm_client) -> None:
    """
    Test that the LLM can correctly answer a basic factual question.
    """
    response: Dict[str, Any] = llm_client.generate("What is the capital of Germany?")
    text_lower: str = response.get("text", "").lower()

    assert "berlin" in text_lower, f"Expected 'Berlin' in response: {response.get('text', '')}"


def test_maintains_context_across_turns(llm_client) -> None:
    """
    Test that the LLM retains context in a multi-turn conversation.
    """
    prompt = (
        "User: My name is Becky\n"
        "Assistant: Hi, Becky\n"
        "User: What is my name?"
    )
    response: Dict[str, Any] = llm_client.generate(prompt)
    text_lower: str = response.get("text", "").lower()

    assert "becky" in text_lower, f"Failed to recall context: {response.get('text', '')}"
