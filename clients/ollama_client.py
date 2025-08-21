import requests, time
from typing import Dict, Any
from .baseclient import BaseLLMClient

class OllamaClient(BaseLLMClient):
    def __init__(self, model: str = "tinyllama", host: str = "localhost", port: int = 11434):
        self.model = model
        self.endpoint = f"http://{host}:{port}/api/generate"

    def _post(self, payload: Dict[str, Any]):
        return requests.post(self.endpoint, json=payload, timeout=15)

    def is_model_available(self) -> bool:
        try:
            resp = self._post({"model": self.model, "prompt": "test", "stream": False})
            return resp.status_code == 200
        except:
            return False

    def generate(self, prompt: str, temperature: float = 0.8, max_tokens: int = 1000) -> Dict[str, Any]:
        start = time.time()
        try:
            resp = self._post({"model": self.model, "prompt": prompt, "stream": False,
                               "temperature": temperature, "max_tokens": max_tokens})
        except Exception as e:
            return {"text": "", "error": str(e), "latency": 0.0, "prompt_tokens": 0, "completion_tokens": 0}
        
        latency = time.time() - start
        try:
            text = resp.json().get("response", "")
        except:
            text = ""

        return {"text": text, "latency": latency,
                "prompt_tokens": len(prompt.split()), "completion_tokens": len(text.split()),
                "error": "" if text else "No response"}
