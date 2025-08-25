import statistics
from typing import Dict, Any
import concurrent.futures
import pytest

# Helper function to fail on API error
def assert_no_api_error(response: Dict[str, Any]):
    if response.get("error"):
        pytest.fail(f"LLM request failed: {response['error']}")


@pytest.mark.performance
def test_average_throughput(llm_client) -> None:
    """
    Test average throughput over multiple runs.
    """
    throughputs = []
    for _ in range(5):
        response: Dict[str, Any] = llm_client.generate("Why is the sky blue?")
        assert_no_api_error(response)

        completion_tokens = response["completion_tokens"]
        latency = response["latency"]
        throughput = completion_tokens / latency if latency > 0 else 0
        throughputs.append(throughput)

    avg_throughput = statistics.mean(throughputs)
    print(f"Average throughput: {avg_throughput:.2f} tokens/sec")
    assert avg_throughput >= 5, f"Average throughput too low: {avg_throughput:.2f} tokens/sec"


@pytest.mark.performance
def test_token_efficiency(llm_client) -> None:
    """
    Ensure the model does not produce an excessive number of tokens for simple prompts.
    """
    response = llm_client.generate("What is 2+2?")
    assert_no_api_error(response)

    completion_tokens = response["completion_tokens"]
    assert completion_tokens <= 5, f"Too many tokens for simple output: {completion_tokens}"


@pytest.mark.performance
def test_latency(llm_client) -> None:
    """
    Test that the LLM responds within an acceptable latency threshold.
    """
    response = llm_client.generate("Say hello in one sentence.")
    assert_no_api_error(response)

    latency = response["latency"]
    print(f"Latency: {latency:.2f} seconds")
    assert latency <= 2.0, f"Latency too high: {latency:.2f}s"


@pytest.mark.performance
def test_concurrent_requests(llm_client) -> None:
    """
    Test that multiple requests can run in parallel without excessive slowdown.
    """
    prompts = ["Write one sentence about AI."] * 5

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(llm_client.generate, prompt) for prompt in prompts]
        results = [f.result() for f in futures]

    # Fail fast if any request failed
    for r in results:
        assert_no_api_error(r)

    latencies = [r["latency"] for r in results]
    print(f"Concurrent latencies: {[round(l, 2) for l in latencies]} seconds")

    avg_latency = sum(latencies) / len(latencies)
    assert avg_latency < 3.0, f"Average latency too high under concurrency: {avg_latency:.2f}s"

@pytest.mark.performance
def test_flight_lookup_latency(llm_client) -> None:
    """
    Ensure flight search responses are fast.
    """
    response: Dict[str, Any] = llm_client.generate("List flights from New York to London on July 15")
    assert_no_api_error(response)

    latency = response["latency"]
    print(f"Flight lookup latency: {latency:.2f} seconds")
    assert latency <= 3.0, f"Flight lookup too slow: {latency:.2f}s"


@pytest.mark.performance
def test_visa_lookup_latency(llm_client) -> None:
    """
    Ensure visa requirement responses are fast.
    """
    response: Dict[str, Any] = llm_client.generate(
        "What documents are required for an Indian citizen to fly to the US?"
    )
    assert_no_api_error(response)

    latency = response["latency"]
    print(f"Visa lookup latency: {latency:.2f} seconds")
    assert latency <= 2.0, f"Visa lookup too slow: {latency:.2f}s"
