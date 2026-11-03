# Week 3: Graduate-Level Problem Set — LLM Observability, OpenTelemetry Tracing & G-Eval Scorer

Welcome to the Month 5 Week 3 Problem Set. This suite evaluates your mastery of enterprise LLMOps observability: OpenTelemetry (OTel) GenAI semantic conventions, distributed trace span generation, continuous evaluation via G-Eval logit expectation scoring, token throughput metrics, and Prometheus counter instrumentation.

---

## Tier 1: Core Mechanics & OpenTelemetry GenAI Semantic Conventions

### Exercise 1: OpenTelemetry GenAI Trace Span Context Manager
**Concepts**: Distributed tracing, trace ID / span ID propagation, GenAI semantic conventions (`gen_ai.request.model`, `gen_ai.usage.prompt_tokens`, `gen_ai.response.finish_reasons`).

#### Problem Specification
Standard application performance monitoring (APM) fails to capture LLM-specific operational context (token usage, temperature, prompt template versions):
1. Implement a Python context manager `GenAISpan(tracer, model_name, prompt)`:
   - Starts an active OpenTelemetry span following standard `gen_ai.*` naming conventions.
   - Sets attributes: `gen_ai.system = "openai"`, `gen_ai.request.model = model_name`, `gen_ai.request.temperature = 0.7`.
   - Records token usage upon exit: `gen_ai.usage.prompt_tokens`, `gen_ai.usage.completion_tokens`.
   - Captures exceptions and sets span status to `StatusCode.ERROR`.
2. Demonstrate trace span propagation across a 2-stage agent pipeline: Retrieval Span $\to$ LLM Generation Span.

#### Verification Solution
```python
from contextlib import contextmanager
import time
from typing import Dict, Any, Generator

class MockSpan:
    def __init__(self, name: str):
        self.name = name
        self.attributes: Dict[str, Any] = {}
        self.status = "OK"
        self.error: Exception | None = None
        self.duration_ms: float = 0.0
        self.start_time: float = 0.0

    def set_attribute(self, key: str, value: Any):
        self.attributes[key] = value

    def set_status(self, status: str, error: Exception | None = None):
        self.status = status
        self.error = error

@contextmanager
def gen_ai_span(name: str, model_name: str, prompt: str) -> Generator[MockSpan, None, None]:
    span = MockSpan(name)
    span.start_time = time.perf_counter()
    span.set_attribute("gen_ai.system", "openai")
    span.set_attribute("gen_ai.request.model", model_name)
    span.set_attribute("gen_ai.prompt_preview", prompt[:50])
    try:
        yield span
    except Exception as e:
        span.set_status("ERROR", error=e)
        raise
    finally:
        span.duration_ms = (time.perf_counter() - span.start_time) * 1000.0

# Verification Run
if __name__ == "__main__":
    with gen_ai_span("llm_completion", "llama-3-70b", "Analyze Q3 revenue report") as span:
        time.sleep(0.02)
        # Record tokens
        span.set_attribute("gen_ai.usage.prompt_tokens", 45)
        span.set_attribute("gen_ai.usage.completion_tokens", 120)
        
    print("Span Attributes Recorded:")
    for k, v in span.attributes.items():
        print(f"  {k}: {v}")
    print(f"Span Status: {span.status} | Duration: {span.duration_ms:.2f} ms")
    assert span.attributes["gen_ai.request.model"] == "llama-3-70b"
    assert span.attributes["gen_ai.usage.completion_tokens"] == 120
```

---

## Tier 2: Continuous Evaluation & G-Eval Logit Expectation Scoring

### Exercise 2: Continuous G-Eval Expected Score Calculation
**Concepts**: LLM-as-a-Judge, probability normalization over discrete score tokens, expectation formula $E[S] = \sum s \cdot p(s)$.

#### Problem Specification
Naive LLM-as-a-judge prompts ask the model to emit a single integer score (e.g. "Rate 1-5"), which suffers from severe stochastic variance. G-Eval (Liu et al. 2023) extracts the output token logits for score tokens `"1"`, `"2"`, `"3"`, `"4"`, `"5"` and computes the mathematical expectation:
$$E[\text{Score}] = \sum_{s=1}^5 s \times \frac{\exp(L_s)}{\sum_{j=1}^5 \exp(L_j)}$$
1. Implement `compute_geval_expectation(score_logits: Dict[int, float]) -> float`.
2. Given raw logit values for a coherence evaluation:
   - `"1"`: $-2.5$, `"2"`: $-1.2$, `"3"`: $0.5$, `"4"`: $2.1$, `"5"`: $1.8$.
3. Compute the normalized probabilities and the expected continuous quality score.

#### Verification Solution
```python
import numpy as np
from typing import Dict

def compute_geval_expectation(score_logits: Dict[int, float]) -> float:
    scores = np.array(list(score_logits.keys()), dtype=float)
    logits = np.array(list(score_logits.values()), dtype=float)
    
    # Stable softmax
    logits_max = np.max(logits)
    exp_logits = np.exp(logits - logits_max)
    probs = exp_logits / np.sum(exp_logits)
    
    expected_score = float(np.sum(scores * probs))
    return expected_score

# Verification Run
if __name__ == "__main__":
    sample_logits = {1: -2.5, 2: -1.2, 3: 0.5, 4: 2.1, 5: 1.8}
    expected_val = compute_geval_expectation(sample_logits)
    print(f"Computed Continuous G-Eval Score: {expected_val:.4f} (out of 5.0)")
    assert 3.5 < expected_val < 4.5
    print("G-Eval Expectation calculation verified with zero discrete rounding distortion.")
```

---

## Tier 3: Production Prometheus Metrics Instrumentation

### Exercise 3: Prometheus Metrics Exporter for Token Velocity and Failure Rates
**Concepts**: Prometheus Counters, Histograms, token consumption velocity, latency percentiles.

#### Problem Specification
1. In high-concurrency LLM gateways, operations teams require real-time Prometheus gauges:
   - `llm_requests_total`: Counter partitioned by `model` and `status`.
   - `llm_token_count_total`: Counter partitioned by `type` (`prompt` vs `completion`).
   - `llm_latency_seconds`: Histogram tracking generation latency distribution.
2. Build an instrumentation wrapper `InstrumentedLLMGateway` tracking these metrics across 50 simulated calls.

#### Verification Solution
```python
from collections import defaultdict
import numpy as np

class MockPrometheusRegistry:
    def __init__(self):
        self.counters = defaultdict(float)
        self.histograms = defaultdict(list)

    def inc_counter(self, metric: str, value: float = 1.0, labels: dict = None):
        key = (metric, tuple(sorted((labels or {}).items())))
        self.counters[key] += value

    def observe_histogram(self, metric: str, value: float, labels: dict = None):
        key = (metric, tuple(sorted((labels or {}).items())))
        self.histograms[key].append(value)

# Verification Run
if __name__ == "__main__":
    registry = MockPrometheusRegistry()
    
    # Simulate 50 requests
    for i in range(50):
        status = "200" if i % 10 != 0 else "500"
        tokens = np.random.randint(50, 200)
        latency = np.random.uniform(0.1, 0.8)
        
        registry.inc_counter("llm_requests_total", labels={"model": "mistral-7b", "status": status})
        registry.inc_counter("llm_token_count_total", value=tokens, labels={"model": "mistral-7b", "type": "completion"})
        registry.observe_histogram("llm_latency_seconds", value=latency, labels={"model": "mistral-7b"})
        
    success_key = ("llm_requests_total", (("model", "mistral-7b"), ("status", "200")))
    error_key = ("llm_requests_total", (("model", "mistral-7b"), ("status", "500")))
    
    print(f"Total Successful Requests: {registry.counters[success_key]}")
    print(f"Total Failed Requests:     {registry.counters[error_key]}")
    assert registry.counters[success_key] == 45
    assert registry.counters[error_key] == 5
    print("Prometheus metrics instrumentation verified successfully!")
```
