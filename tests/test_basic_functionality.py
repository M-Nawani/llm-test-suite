from typing import Dict, Any

import pytest

# Helper to fail on API error
def assert_no_api_error(response: Dict[str, Any]):
    if response.get("error"):
        pytest.fail(f"LLM request failed: {response['error']}")

@pytest.mark.functionality
def test_generate_non_empty_response(llm_client) -> None:
    """
    Test that the LLM generates a non-empty, sufficiently long response
    to a simple prompt.
    """
    response: Dict[str, Any] = llm_client.generate("How are you?")
    assert_no_api_error(response)

    text: str = response["text"].strip()
    assert text, "Response text is empty."
    assert len(text) > 10, f"Response too short: {len(text)} characters."


@pytest.mark.functionality
def test_follow_simple_instruction(llm_client) -> None:
    """
    Test that the LLM follows a simple instruction by naming at least
    one color from a predefined list.
    """
    response: Dict[str, Any] = llm_client.generate(
        "Name three colors. Just list the colors."
    )
    assert_no_api_error(response)

    text_lower: str = response["text"].strip().lower()
    colors = {
        "violet", "indigo", "blue", "green", "yellow",
        "orange", "red", "white", "black", "grey"
    }
    found_colors = {color for color in colors if color in text_lower}
    assert found_colors, f"No expected colors found in: {response['text']}"


@pytest.mark.functionality
def test_answer_simple_fact_question(llm_client) -> None:
    """
    Test that the LLM can correctly answer a basic factual question.
    """
    response: Dict[str, Any] = llm_client.generate("What is the capital of Germany?")
    assert_no_api_error(response)

    text_lower: str = response["text"].strip().lower()
    assert "berlin" in text_lower, f"Expected 'Berlin' in response: {response['text']}"


@pytest.mark.functionality
def test_estav_requirement(llm_client) -> None:
    """
    Test that the LLM can correctly answer a basic visa realted question.
    """
    response: Dict[str, Any] = llm_client.generate("Do EU citizens need a visa to visit Puerto Rico?")
    assert_no_api_error(response)

    text_lower: str = response["text"].strip().lower()
    assert "esta" in text_lower, f"Expected 'ESTA' in response: {response['text']}"


@pytest.mark.functionality
def test_visa_requirement(llm_client) -> None:
    """
    Test that the LLM can correctly answer a basic visa realted question.
    """
    response: Dict[str, Any] = llm_client.generate("Do Indian citizens require a visa for Bhutan")
    assert_no_api_error(response)

    text_lower: str = response["text"].strip().lower()
    assert "entry permit" in text_lower, f"Expected 'Entry permit' in response: {response['text']}"


@pytest.mark.functionality
def test_currency_of_japan(llm_client) -> None:
    """
    Test that the LLM can correctly answer a currency related question
    """
    response: Dict[str, Any] = llm_client.generate("What currency is used in Japan?")
    assert_no_api_error(response)

    text_lower: str = response["text"].strip().lower()
    assert "japanese yen" in text_lower, f"Expected 'Japanese Yen' in response: {response['text']}"


@pytest.mark.functionality
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
    assert_no_api_error(response)

    text_lower: str = response["text"].strip().lower()
    assert "becky" in text_lower, f"Failed to recall context: {response['text']}"
