import pytest
import json
from typing import Dict, Any
import re
from llmtestclient import llm_client

def test_closed_domain_factual_accuracy(llm_client) -> None:
    """
    Test that the model returns correct facts for well-known queries.
    Accepts both 'Shakespeare' and 'William Shakespeare' as correct.
    """
    fact = "Who wrote the play Hamlet?"
    valid_answers = ["william shakespeare", "shakespeare"]

    response: Dict[str, Any] = llm_client.generate(fact)
    answer = response["text"].strip().lower()

    assert any(valid in answer for valid in valid_answers), (
        f"Expected one of {valid_answers} in answer, got: {answer}"
    )


def test_unknown_fact_admission(llm_client) -> None:
    """
    Test that the model admits when it doesn't know an obscure fact.
    """
    obscure_q = "Who was the second person to set foot on Mars?"
    response: Dict[str, Any] = llm_client.generate(obscure_q)
    answer = response["text"].lower()
    assert any(phrase in answer for phrase in ["i don't know", "unknown", "no record"]), (
        f"Possible hallucination: {answer}"
    )


def test_numerical_precision(llm_client) -> None:
    """
    Test that the model answers basic arithmetic correctly.
    """
    prompt = "What is 17 multiplied by 19?"
    response: Dict[str, Any] = llm_client.generate(prompt)
    answer = re.findall(r'\d+', response["text"])
    numbers = [int(num) for num in answer if num.isdigit()]
    assert 323 in numbers, f"Expected 323, got: {response['text']}"


def test_entity_consistency_in_context(llm_client) -> None:
    """
    Test that the model maintains entity details over a conversation.
    """
    intro = "My friend's name is Alex. Alex is 27 years old."
    question = "How old is Alex?"
    llm_client.generate(intro)
    response: Dict[str, Any] = llm_client.generate(question)
    assert "27" in response["text"], (
        f"Expected Alex to still be 27, got: {response['text']}"
    )
