# 🧪 LLM Evaluation Test Suite

This repository contains a **Pytest-based testing framework** for evaluating **Large Language Models (LLMs)** running on [Ollama](https://ollama.ai/).  
It focuses on **robustness**, **performance**, **context handling**, and **hallucination detection**.

## 📂 Project structure

```plaintext
llmtestingwithpython/
├── clients/
│   ├── baseclient.py                              # Base LLM client interface
│   ├── ollama_client.py                           # Ollama LLM client implementation
│   └── __init__.py
├── tests/
│   ├── test_basic_functionality.py                # Basic behavior & sanity checks
│   ├── test_context_learning.py                   # Context retention & in-context learning
│   ├── test_context_processing_generation.py      # Structured output & JSON validation
│   ├── test_performance.py                        # Throughput & latency benchmarks
│   ├── test_robustness.py                         # Prompt injection & logical consistency
│   ├── test_hallucination.py                      # Hallucination detection & factual tests
│   └── __init__.py
├── __pycache__/                                   # Python cache files (ignored in git)
├── .pytest_cache/                                 # Pytest cache
├── .git/                                         # Git repository data
├── venv/                                         # Python virtual environment
├── conftest.py                                   # Pytest fixtures
├── pytest.ini                                    # Pytest configuration
├── requirements.txt                              # Python dependencies
├── README.md                                     # Project README
├── LICENSE                                       # License file
└── .gitignore                                    # Git ignore rules
```

## 🚀 Features

- **Model Availability Check** – Skips tests if the LLM is unavailable.
- **Context Learning Tests** – Checks if the model retains information across prompts.
- **Structured Output Tests** – Validates JSON and list generation.
- **Performance Metrics** – Measures latency, token throughput.
- **Robustness Checks** – Tests against prompt injection, nonsensical queries, and logical consistency.
- **Hallucination Detection** – Ensures factual accuracy in closed domains.


## 📦 Installation

```bash
# Clone the repo
git clone https://github.com/m-nawani/llm-test-suite.git
cd llm-test-suite
```

# Install dependencies
pip install -r requirements.txt

## ⚙️ Configuration

You can configure the **OllamaTestClient** in `llmtestclient.py` to change the model, host, or port.

| Setting   | Default Value | Description |
|-----------|--------------|-------------|
| `model`   | `tinyllama`  | The Ollama model to use (e.g., `mistral`, `llama2-7b`) |
| `host`    | `localhost`  | The API server hostname |
| `port`    | `11434`      | The API server port |

Example:
```python
from llmtestclient import OllamaTestClient

# Use a different model and server
client = OllamaTestClient(model="mistral", host="localhost", port=11434)
```

## 🧪 Running All Tests

- To run the full LLM test suite:

```bash
pytest -v
```

- To run only specific test files:

```bash
pytest tests/test_performance.py
pytest tests/test_robustness.py
```

- To run a single test function

```bash
pytest tests/test_robustness.py::test_prompt_injection_resilience -q
```

- To run tests based on markers

```bash
pytest -m robustness
```
- To view the allure reports

```bash
allure generate allure-results -o allure-report --clean
```

## 📌 Notes

- Designed for lightweight Ollama models (`tinyllama`, `llama2-7b`, `mistral`) but adaptable for remote endpoints.  
- Mock or sandbox external factual sources to keep hallucination tests deterministic.  
- Mark heavy benchmarks as **slow** to keep the default test run fast.  
- Extend the suite with domain-specific tests and schema validation for structured outputs.

## 🚧 Work in Progress

- Containerize the application for consistent local and CI environments.  
- Add CI/CD pipeline to automate:
  - Build
  - Test
  - Deployment
- Integrate with existing monitoring and logging setup.


## 📜 License

This project is licensed under the [MIT License](LICENSE).





