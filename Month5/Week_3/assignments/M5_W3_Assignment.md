# Week 3 Production Engineering Assignment: Enterprise LLM Observability, OpenTelemetry Tracing & Continuous G-Eval Suite

## Executive Overview & Architectural Motivation
Deploying autonomous agents and LLM applications into production without deep observability is a liability. Model hallucination, prompt injection attacks, catastrophic latency spikes from long-context prefilling, and unexpected token billing explosions occur silently behind standard HTTP 200 responses.

In this assignment, you will architect a production-grade **LLM Observability & Continuous Evaluation Engine** combining:
1. **OpenTelemetry (OTel) Distributed Tracing**: Instrumenting prompts, model names, latency, and token consumption using standard GenAI semantic conventions.
2. **Prometheus Metrics Exporter**: Tracking token consumption velocity, error rates, and latency percentiles.
3. **Continuous G-Eval Evaluation Pipeline**: Logit-expectation scoring of factual correctness, answer relevance, and safety.
4. **Structured JSONL Audit Logging**: Capturing complete prompt/completion traces for regulatory compliance.

---

## Technical Specifications & Architecture Diagram

```
                 Incoming Application Request -> LLM Gateway
                                       │
                                       ▼
                 ┌───────────────────────────────────────────────────────┐
                 │          OpenTelemetry Trace Span Interceptor         │
                 │   - Injects Trace ID & Span Context                   │
                 │   - Records gen_ai.request.model & temperature        │
                 └─────────────────────┬─────────────────────────────────┘
                                       │
                                       ▼
                 ┌───────────────────────────────────────────────────────┐
                 │               Prometheus Metrics Hook                 │
                 │   - llm_requests_total (counter)                      │
                 │   - llm_token_count_total (counter: prompt/completion)│
                 │   - llm_latency_seconds (histogram)                   │
                 └─────────────────────┬─────────────────────────────────┘
                                       │
                                       ▼
                 ┌───────────────────────────────────────────────────────┐
                 │           Continuous G-Eval Evaluation Judge          │
                 │   - Mathematical Expectation over Token Logits        │
                 │   - Threshold Check: Score < 3.5 -> Alert Trigger     │
                 └─────────────────────┬─────────────────────────────────┘
                                       │
                                       ▼
                 Structured Audit Logs & Operational Prometheus Gauges
```

---

## Detailed Deliverable Requirements

### 1. OpenTelemetry GenAI Context Wrapper (`tracing.py`)
Implement `ObservableLLMClient`:
- Wraps LLM invocation inside OpenTelemetry spans.
- Adheres strictly to OTel GenAI Semantic Conventions:
  - `gen_ai.system`: e.g. `"openai"`, `"anthropic"`, `"vllm"`
  - `gen_ai.request.model`: e.g. `"llama-3-8b-instruct"`
  - `gen_ai.usage.prompt_tokens`: integer count
  - `gen_ai.usage.completion_tokens`: integer count
  - `gen_ai.response.finish_reasons`: list of strings (e.g. `["stop"]`)

### 2. Continuous G-Eval Judge Engine (`geval.py`)
Implement `GEvalScorer`:
- Evaluates output text across criteria: *Coherence*, *Groundedness*, *Safety*.
- Extracts score token probabilities and computes continuous expectation:
  $$E[S] = \sum_{s=1}^5 s \cdot P(S=s)$$
- Emits real-time alerts if any production sample scores below $3.5 / 5.0$.

### 3. Prometheus Metric Exporter (`metrics.py`)
- Expose an HTTP endpoint `/metrics` format tracking:
  - `llm_token_throughput_per_sec`
  - `llm_cost_estimated_usd`
  - `llm_request_duration_seconds`

---

## Starter Code Template & Submission Skeleton

```python
"""
llm_observability_suite.py - Production LLMOps Observability Suite
"""
import time
from dataclasses import dataclass
from typing import Dict, Any, List
import numpy as np

@dataclass
class LLMTraceEvent:
    trace_id: str
    span_id: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    duration_ms: float
    geval_score: float
    status: str

class ProductionLLMObservabilityEngine:
    def __init__(self, quality_threshold: float = 3.5):
        self.quality_threshold = quality_threshold
        self.traces: List[LLMTraceEvent] = []
        self.total_tokens_consumed = 0
        self.alert_count = 0

    def compute_geval_score(self, mock_logits: Dict[int, float]) -> float:
        scores = np.array(list(mock_logits.keys()))
        logits = np.array(list(mock_logits.values()))
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        return float(np.sum(scores * probs))

    def record_llm_call(self, trace_id: str, model: str, prompt: str, completion: str,
                        prompt_toks: int, comp_toks: int, latency_ms: float, 
                        judge_logits: Dict[int, float]) -> LLMTraceEvent:
        geval = self.compute_geval_score(judge_logits)
        status = "HEALTHY" if geval >= self.quality_threshold else "ALERT_LOW_QUALITY"
        
        if status != "HEALTHY":
            self.alert_count += 1
            
        event = LLMTraceEvent(
            trace_id=trace_id,
            span_id=f"span_{len(self.traces)+1}",
            model=model,
            prompt_tokens=prompt_toks,
            completion_tokens=comp_toks,
            duration_ms=latency_ms,
            geval_score=geval,
            status=status
        )
        self.traces.append(event)
        self.total_tokens_consumed += (prompt_toks + comp_toks)
        return event

# Verification Run
if __name__ == "__main__":
    engine = ProductionLLMObservabilityEngine(quality_threshold=3.5)
    
    # 1. High quality completion
    good_logits = {1: -5.0, 2: -3.0, 3: 0.5, 4: 2.5, 5: 3.0}
    ev1 = engine.record_llm_call(
        "tx_101", "llama-3-8b", "What is gradient descent?", "Gradient descent is an optimization...",
        prompt_toks=15, comp_toks=50, latency_ms=120.0, judge_logits=good_logits
    )
    
    # 2. Degraded / hallucinated completion
    bad_logits = {1: 3.5, 2: 2.0, 3: 0.1, 4: -2.0, 5: -4.0}
    ev2 = engine.record_llm_call(
        "tx_102", "llama-3-8b", "Explain photosynthesis", "Photosynthesis is when rocks absorb...",
        prompt_toks=12, comp_toks=35, latency_ms=95.0, judge_logits=bad_logits
    )
    
    print(f"Trace 1 Status: {ev1.status} | G-Eval Score: {ev1.geval_score:.2f}")
    print(f"Trace 2 Status: {ev2.status} | G-Eval Score: {ev2.geval_score:.2f}")
    print(f"Total Tokens Tracked: {engine.total_tokens_consumed} | Total Alerts: {engine.alert_count}")
    
    assert ev1.status == "HEALTHY"
    assert ev2.status == "ALERT_LOW_QUALITY"
    assert engine.alert_count == 1
    print("Observability and continuous evaluation engine verified successfully!")
```

---

## Grading Rubric (100 Points Total)

| Component | Points | Criteria |
| :--- | :--- | :--- |
| **OpenTelemetry GenAI Conventions** | 30 | Accurate span attributes, context propagation, error tagging. |
| **G-Eval Logit Expectation Engine** | 30 | Softmax probability distribution over discrete score tokens, expectation formula. |
| **Prometheus Metrics & Velocity Tracking** | 20 | Monotonically increasing counters, latency histograms, token gauges. |
| **Alerting & Quality Guardrails** | 10 | Real-time threshold breach detection and audit payload formatting. |
| **Verification & Reproducibility** | 10 | Complete automated test suite asserting metric increments and score values. |
