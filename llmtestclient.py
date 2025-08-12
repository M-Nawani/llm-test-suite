import requests
import time
import pytest
from typing import Dict, Any, Optional


class OllamaTestClient:
    """
    Client for testing Ollama-based LLM models via the /api/generate endpoint.
    """

    def __init__(self, model: str = "tinyllama", host: str = "localhost", port: int = 11434) -> None:
        self.model = model
        self.endpoint = f"http://{host}:{port}/api/generate"

    def _post(self, payload: Dict[str, Any]) -> requests.Response:
        """
        Internal helper for making POST requests to the endpoint.
        Raises an exception if the request fails.
        """
        try:
            response = requests.post(self.endpoint, json=payload, timeout=15)
            return response
        except requests.RequestException as e:
            raise ConnectionError(f"Failed to connect to {self.endpoint}: {e}")

    def is_model_available(self) -> bool:
        """
        Check whether the model is available and responding.
        """
        try:
            response = self._post({
                "model": self.model,
                "prompt": "test",
                "stream": False
            })
            return response.status_code == 200
        except Exception as e:
            print(f"[WARNING] Model availability check failed: {e}")
            return False

    def generate(
        self,
        prompt: str,
        temperature: float = 0.8,
        max_tokens: int = 1000
    ) -> Dict[str, Any]:
        """
        Generate text from the LLM.
        """
        start_time = time.time()

        try:
            response = self._post({
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "temperature": temperature,
                "max_tokens": max_tokens
            })
        except Exception as e:
            return {
                "error": f"Request failed: {e}",
                "text": "",
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "latency": 0.0
            }

        latency = time.time() - start_time

        if response.status_code != 200:
            return {
                "error": f"API Error {response.status_code}: {response.text}",
                "text": "",
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": 0,
                "latency": latency
            }

        try:
            result: Dict[str, Any] = response.json()
        except ValueError as e:
            return {
                "error": f"Invalid JSON in API response: {e}",
                "text": "",
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": 0,
                "latency": latency
            }

        text: str = result.get("response", "")

        return {
            "text": text,
            "prompt_tokens": len(prompt.split()),
            "completion_tokens": len(text.split()),
            "latency": latency
        }


@pytest.fixture
def llm_client() -> OllamaTestClient:
    """
    Pytest fixture that provides a ready-to-use OllamaTestClient instance.
    Skips tests if the model is unavailable.
    """
    client = OllamaTestClient(model="tinyllama")
    if not client.is_model_available():
        pytest.skip(f"Model '{client.model}' is not available at {client.endpoint}")
    return client
