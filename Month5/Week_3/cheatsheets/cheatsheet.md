# Month 5, Week 3 Cheatsheet: LLM Observability, OpenTelemetry & Continuous Evaluation

---

## 1. OpenTelemetry (OTel) GenAI Semantic Conventions

Standardized tracing attributes for monitoring foundational model invocations:

| Semantic Attribute Key | Type | Description | Example Value |
|:---|:---|:---|:---|
| `gen_ai.system` | `string` | Target inference provider | `"openai"`, `"anthropic"`, `"vllm"` |
| `gen_ai.request.model` | `string` | Target model name | `"llama-3-70b-instruct"` |
| `gen_ai.request.temperature` | `double` | Generation temperature | `0.7` |
| `gen_ai.usage.prompt_tokens` | `int` | Input token count | `124` |
| `gen_ai.usage.completion_tokens` | `int` | Output generated tokens | `48` |
| `gen_ai.response.finish_reasons` | `string[]` | Stop reason array | `["stop"]`, `["length"]` |

---

## 2. Continuous Evaluation: G-Eval Logit Expectation Scoring

Eliminates discrete rounding variance in LLM-as-a-judge evaluations:
$$E[\text{Score}] = \sum_{s=1}^5 s \times P(\text{Token}=s) = \sum_{s=1}^5 s \times \frac{\exp(L_s)}{\sum_{j=1}^5 \exp(L_j)}$$
where $L_s$ is the raw unnormalized logit of score token $s \in \{1, 2, 3, 4, 5\}$.

```python
import numpy as np

def compute_geval_expectation(score_logits: dict[int, float]) -> float:
    scores = np.array(list(score_logits.keys()))
    logits = np.array(list(score_logits.values()))
    
    # Stable softmax
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / np.sum(exp_logits)
    
    return float(np.sum(scores * probs))
```

---

## 3. Production Prometheus Instrumentation

Standard metric exports for enterprise LLM gateways:
```python
from prometheus_client import Counter, Histogram

# 1. Total request count partitioned by status
LLM_REQUESTS = Counter(
    "llm_requests_total", 
    "Total LLM gateway requests", 
    ["model", "status"]
)

# 2. Token consumption velocity
LLM_TOKENS = Counter(
    "llm_token_count_total", 
    "Total tokens consumed", 
    ["model", "type"]  # type: 'prompt' or 'completion'
)

# 3. Request latency distribution
LLM_LATENCY = Histogram(
    "llm_latency_seconds", 
    "End-to-end request latency", 
    ["model"],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)
```
