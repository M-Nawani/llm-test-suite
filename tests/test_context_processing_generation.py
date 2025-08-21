import re
import json
import pytest
from typing import Dict, Any

# Helpers
def assert_min_numbered_items(text: str, min_items: int = 3):
    numbered_lines = re.findall(r'^\d+\.\s+\S+', text, flags=re.MULTILINE)
    assert len(numbered_lines) >= min_items, (
        f"Expected at least {min_items} numbered items, found {len(numbered_lines)}.\nOutput:\n{text}"
    )

def assert_valid_json(text: str, expected_fields: Dict[str, Any]):
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as e:
        raise AssertionError(f"Output is not valid JSON: {e}\nOutput:\n{text}")

    if isinstance(parsed, list):
        assert parsed, f"Expected non-empty list, got {parsed}"
        parsed = parsed[0]

    assert isinstance(parsed, dict), f"Expected JSON object, got {type(parsed)}"

    for field, value in expected_fields.items():
        assert str(parsed.get(field)) == str(value), f"Expected {field}='{value}', got {parsed.get(field)}"


def assert_summary_contains_keywords(summary: str, keywords: list[str]):
    missing_keywords = [kw for kw in keywords if kw.lower() not in summary.lower()]
    assert not missing_keywords, f"Missing key info in summary: {missing_keywords}"


@pytest.mark.content_generation
def test_numbered_list_output(llm_client) -> None:
    response: Dict[str, Any] = llm_client.generate(
        "List 3 fruits. The output should be a numbered list."
    )
    if response.get("error"):
        pytest.fail(f"LLM request failed: {response['error']}")

    text = response["text"].strip()
    assert_min_numbered_items(text, min_items=3)


@pytest.mark.content_generation
def test_json_output(llm_client) -> None:
    response: Dict[str, Any] = llm_client.generate(
        "Create a JSON file with name: 'John' and age: '33'."
    )
    if response.get("error"):
        pytest.fail(f"LLM request failed: {response['error']}")

    text = response["text"].strip()
    assert_valid_json(text, expected_fields={"name": "John", "age": "33"})


@pytest.mark.content_generation
def test_summarization_quality(llm_client) -> None:
    original_text = (
        "Albert Einstein was a German-born theoretical physicist who developed the theory "
        "of relativity, one of the two pillars of modern physics. His work is also known "
        "for its influence on the philosophy of science. He won the Nobel Prize in Physics in 1921."
    )

    response: Dict[str, Any] = llm_client.generate(
        f"Summarize the following text in 2 sentences:\n\n{original_text}"
    )
    if response.get("error"):
        pytest.fail(f"LLM request failed: {response['error']}")

    summary = response["text"].strip()
    assert len(summary) < len(original_text), "Summary is not shorter than original text."
    assert_summary_contains_keywords(summary, ["Einstein", "physicist", "relativity", "Nobel"])
