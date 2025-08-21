# conftest.py
import pytest
import allure
from functools import wraps
from clients.ollama_client import OllamaClient


def pytest_addoption(parser):
    parser.addoption(
        "--llm", action="store", default="ollama", help="Choose which LLM client to use"
    )


@pytest.fixture
def llm_client(request):
    """
    Fixture that returns the selected LLM client.
    """
    client_name = request.config.getoption("--llm")

    if client_name.lower() == "ollama":
        client = OllamaClient()
    else:
        pytest.fail(f"Unknown LLM client: {client_name}")

    if not client.is_model_available():
        pytest.skip(f"Model '{client_name}' is not available.")

    return client


@pytest.fixture(autouse=True)
def attach_llm_responses(monkeypatch, llm_client):
    """
    Automatically attach LLM responses to Allure for any test using llm_client.
    """
    original_generate = llm_client.generate

    @wraps(original_generate)
    def wrapper(*args, **kwargs):
        response = original_generate(*args, **kwargs)
        if "text" in response and response["text"]:
            allure.attach(
                response["text"],
                name="LLM Response",
                attachment_type=allure.attachment_type.TEXT
            )
        return response

    monkeypatch.setattr(llm_client, "generate", wrapper)
