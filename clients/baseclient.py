from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseLLMClient(ABC):
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate a response from the LLM"""
        pass

    @abstractmethod
    def is_model_available(self) -> bool:
        """Check if the model is available"""
        pass
