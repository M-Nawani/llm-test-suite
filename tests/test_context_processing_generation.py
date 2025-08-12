import re
from typing import Dict, Any
from llmtestclient import llm_client
import json

def test_numbered_list_output(llm_client) -> None:
    """
    Test that the LLM outputs a numbered list when explicitly instructed.
    """
    response: Dict[str, Any] = llm_client.generate(
        "List 3 fruits. The output should be a numbered list"
    )
    text = response["text"].strip()

    # Find lines starting with number + period + space
    numbered_lines = re.findall(r'^\d+\.\s+\S+', text, flags=re.MULTILINE)

    assert len(numbered_lines) >= 3, (
        f"Expected at least 3 numbered items, found {len(numbered_lines)}.\n"
        f"Output:\n{text}"
    )


def test_json_output(llm_client) -> None:
    """
    Test that the LLM generates valid JSON with the expected name and age fields.
    Handles both dict output and list-of-dicts output.
    """
    response: Dict[str, Any] = llm_client.generate(
        "Create a JSON file with name: 'John' and age: '33'"
    )
    text = response["text"].strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as e:
        raise AssertionError(f"Output is not valid JSON: {e}\nOutput:\n{text}")

    # Accept either dict or list-of-dicts
    if isinstance(parsed, list):
        assert parsed, f"Expected non-empty list, got {parsed}"
        parsed = parsed[0]

    assert isinstance(parsed, dict), f"Expected JSON object, got {type(parsed)}"

    assert parsed.get("name") == "John", f"Expected name 'John', got {parsed.get('name')}"
    assert str(parsed.get("age")) == "33", f"Expected age '33', got {parsed.get('age')}"


def test_summarization_quality(llm_client) -> None:
    """
    Test that the LLM summarizes text by shortening it while preserving key information.
    """
    original_text = (
        "Albert Einstein was a German-born theoretical physicist who developed the theory "
        "of relativity, one of the two pillars of modern physics. His work is also known "
        "for its influence on the philosophy of science. He won the Nobel Prize in Physics in 1921."
    )

    response: Dict[str, Any] = llm_client.generate(
        f"Summarize the following text in 2 sentences:\n\n{original_text}"
    )
    summary = response["text"].strip()

    # 1. Check length reduction
    assert len(summary) < len(original_text), (
        f"Summary is not shorter. Lengths: summary={len(summary)}, original={len(original_text)}"
    )

    # 2. Check presence of key facts
    required_keywords = ["Einstein", "physicist", "relativity", "Nobel"]
    missing_keywords = [kw for kw in required_keywords if kw.lower() not in summary.lower()]
    assert not missing_keywords, f"Missing key info in summary: {missing_keywords}"

