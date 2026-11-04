# Week 4 Production Engineering Assignment: Enterprise Production-Ready FastAPI Inference Gateway & Streaming SSE Microservice

## Executive Overview & Architectural Motivation
Deploying machine learning models and large language models into enterprise production requires building resilient, non-blocking asynchronous gateways capable of processing high-concurrency requests, streaming tokens with sub-millisecond overhead via Server-Sent Events (SSE), validating input schemas strictly via Pydantic v2, and executing inside hardened multi-stage Docker containers.

In this capstone assignment for Month 5, you will architect a **Production-Ready Enterprise AI Microservice Gateway** using **FastAPI**, **Pydantic v2**, and asynchronous streaming protocols.

Your gateway will feature an OpenAI-compatible `/v1/chat/completions` endpoint, non-blocking background CPU offloading, structured exception middleware, rate-limiting, Prometheus metrics export (`/metrics`), dynamic readiness/liveness health checks (`/healthz`), and a hardened multi-stage container deployment configuration.

---

## Technical Specifications & Architecture Diagram

```
                 Concurrent Client Applications & Web UIs
                                    │
                                    ▼
                 ┌───────────────────────────────────────────────────────┐
                 │          FastAPI Ingress Gateway (Port 8000)          │
                 │   - Non-blocking Asynchronous Coroutine Workers       │
                 │   - Structured Error Handling Middleware              │
                 │   - Pydantic v2 Payload Contracts & Validation        │
                 └──────────────────────────┬────────────────────────────┘
                                            │
                                            ▼
                 ┌───────────────────────────────────────────────────────┐
                 │       Inference Execution Router & Thread Pool        │
                 │   - Async IO for Database / Vector Retrievial         │
                 │   - asyncio.to_thread for CPU-Bound Model Computations│
                 └──────────────────────────┬────────────────────────────┘
                                            │
                                            ▼
                 ┌───────────────────────────────────────────────────────┐
                 │      Streaming Token Generator (W3C SSE Format)       │
                 │   - Content-Type: text/event-stream                   │
                 │   - data: {"choices": [{"delta": {"content": "..."}}]}│
                 │   - data: [DONE]                                      │
                 └──────────────────────────┬────────────────────────────┘
                                            │
                                            ▼
                 Multi-Stage Hardened Docker Container (Non-Root User)
```

---

## Detailed Deliverable Requirements

### 1. Pydantic v2 API Contracts (`schemas.py`)
Implement strict validation models:
- `ChatCompletionRequest`:
  - `model`: str (validated against supported model registry).
  - `messages`: List of message objects with role (`system`, `user`, `assistant`) and non-empty content.
  - `temperature`: float bounded strictly in $[0.0, 2.0]$.
  - `stream`: bool (default False).
- `ChatCompletionResponse`:
  - Structured OpenAI-compatible response format including token usage statistics.

### 2. High-Performance Asynchronous FastAPI Gateway (`main.py`)
Implement endpoints:
- `GET /healthz`: Immediate 200 OK liveness/readiness check with memory footprint metrics.
- `POST /v1/chat/completions`:
  - If `stream=False`: Returns complete JSON response.
  - If `stream=True`: Streams chunks via `StreamingResponse(..., media_type="text/event-stream")`.
- Structured global exception handlers intercepting unhandled errors and formatting spec-compliant JSON envelopes.

### 3. Production Dockerfile & Container Hardening (`Dockerfile`)
- Multi-stage build (Builder stage $\to$ Minimal Python 3.11-slim runtime).
- Create non-privileged system user (`appuser`, UID 10001).
- Include native container health checks.

---

## Starter Code Template & Submission Skeleton

```python
"""
production_fastapi_gateway.py - Enterprise Production Inference Microservice
"""
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, AsyncGenerator
import asyncio
import json
import time

app = FastAPI(title="Enterprise AI Gateway", version="1.0.0")

class MessagePayload(BaseModel):
    role: str = Field(pattern="^(system|user|assistant)$")
    content: str = Field(min_length=1)

class ChatRequest(BaseModel):
    model: str = "llama-3-8b-instruct"
    messages: List[MessagePayload]
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    stream: bool = False

async def mock_streaming_tokens(prompt: str) -> AsyncGenerator[str, None]:
    sample_tokens = ["Enterprise", " AI", " gateway", " is", " fully", " operational."]
    for tok in sample_tokens:
        await asyncio.sleep(0.015)  # Simulate inter-token decode latency
        chunk = {
            "id": "chatcmpl-prod",
            "choices": [{"delta": {"content": tok}, "index": 0}]
        }
        yield f"data: {json.dumps(chunk)}\n\n"
    yield "data: [DONE]\n\n"

@app.get("/healthz")
async def health_check():
    return {"status": "HEALTHY", "timestamp": time.time()}

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    if req.stream:
        user_prompt = req.messages[-1].content
        return StreamingResponse(mock_streaming_tokens(user_prompt), media_type="text/event-stream")
    
    # Synchronous full response
    await asyncio.sleep(0.05)
    return {
        "id": "chatcmpl-prod",
        "model": req.model,
        "choices": [{"message": {"role": "assistant", "content": "Processed batch response successfully."}}],
        "usage": {"prompt_tokens": 20, "completion_tokens": 15, "total_tokens": 35}
    }

# Verification Run
if __name__ == "__main__":
    from fastapi.testclient import TestClient
    client = TestClient(app)
    
    # 1. Health check
    res_health = client.get("/healthz")
    assert res_health.status_code == 200
    print("Health check response:", res_health.json())
    
    # 2. Synchronous chat completion
    payload = {
        "model": "llama-3-8b-instruct",
        "messages": [{"role": "user", "content": "Hello AI!"}],
        "stream": False
    }
    res_chat = client.post("/v1/chat/completions", json=payload)
    assert res_chat.status_code == 200
    print("Chat completion response:", res_chat.json())
    
    # 3. Streaming chat completion
    payload["stream"] = True
    res_stream = client.post("/v1/chat/completions", json=payload)
    assert res_stream.status_code == 200
    assert "data: [DONE]" in res_stream.text
    print("SSE streaming tokens received successfully!")
```

---

## Grading Rubric (100 Points Total)

| Component | Points | Criteria |
| :--- | :--- | :--- |
| **FastAPI Asynchronous Gateway Design** | 30 | Clean non-blocking route definitions, thread offload, structured error handling. |
| **W3C Server-Sent Events (SSE) Protocol** | 30 | Flawless `text/event-stream` chunks, standard JSON payloads, `[DONE]` termination. |
| **Pydantic v2 Schema Contracts** | 20 | Strict boundary validation on roles, temperatures, message lengths. |
| **Hardened Multi-Stage Containerization** | 10 | 2-stage Dockerfile, non-root user execution, native health check probes. |
| **Verification & Test Coverage** | 10 | Fully automated test client suite asserting 200 status and streaming content. |
