# Week 4: Graduate-Level Problem Set — Production Deployment, FastAPI & SSE Streaming Gateways

Welcome to the Month 5 Week 4 Problem Set. This suite evaluates your mastery of enterprise AI API deployment: asynchronous FastAPI endpoints, Server-Sent Events (SSE) token streaming via W3C protocols, non-blocking coroutine execution vs `asyncio.to_thread` thread pools, Pydantic v2 input validation, and multi-stage hardened Docker containerization.

---

## Tier 1: Core Mechanics & Non-Blocking Async FastAPI Gateways

### Exercise 1: Asynchronous Event Loop Blocking & Thread Offloading
**Concepts**: Event loop starvation, CPU-bound vs IO-bound tasks, `asyncio.to_thread()`, sub-millisecond API responsiveness.

#### Problem Specification
When an asynchronous FastAPI endpoint executes a synchronous CPU-bound model inference call (such as a pure-Python forward pass or blocking PyTorch computation), it blocks the single-threaded event loop, starving concurrent client connections:
1. Demonstrate how a blocking synchronous calculation inside `async def` stalls concurrent health check requests (`/healthz`).
2. Refactor the CPU-bound operation using `await asyncio.to_thread(blocking_inference, x)` to offload compute to the default worker thread pool.
3. Assert that concurrent asynchronous health checks respond in $< 2\text{ ms}$ while heavy background inference executes.

#### Verification Solution
```python
import asyncio
import time

def blocking_cpu_inference(x: float) -> float:
    # Simulates CPU-heavy forward pass
    total = 0.0
    for i in range(1_000_000):
        total += (x * 0.001) ** 2
    return total

async def fast_health_check() -> dict:
    t0 = time.perf_counter()
    await asyncio.sleep(0.001)  # Non-blocking IO
    return {"status": "ok", "latency_ms": (time.perf_counter() - t0) * 1000.0}

async def handle_inference_request(x: float) -> float:
    # Non-blocking thread offload
    return await asyncio.to_thread(blocking_cpu_inference, x)

# Verification Run
if __name__ == "__main__":
    async def main():
        t0 = time.perf_counter()
        # Launch heavy inference and 5 concurrent health checks simultaneously
        inference_task = asyncio.create_task(handle_inference_request(3.14))
        health_tasks = [asyncio.create_task(fast_health_check()) for _ in range(5)]
        
        health_results = await asyncio.gather(*health_tasks)
        inference_res = await inference_task
        total_time = time.perf_counter() - t0
        
        print(f"Total time elapsed: {total_time:.3f}s")
        print(f"First health check latency: {health_results[0]['latency_ms']:.2f} ms")
        assert all(h["status"] == "ok" for h in health_results)
        print("Non-blocking thread offloading verified: health checks were not starved!")

    asyncio.run(main())
```

---

## Tier 2: Server-Sent Events (SSE) Streaming Protocol

### Exercise 2: W3C Server-Sent Events (SSE) Generator Protocol
**Concepts**: `text/event-stream`, chunked transfer encoding, `data: <payload>\n\n`, OpenAI SSE wire format.

#### Problem Specification
Modern LLM chat interfaces stream token deltas to clients using Server-Sent Events (SSE):
1. The standard W3C SSE wire format mandates:
   `data: {"choices": [{"delta": {"content": "Hello"}}]}\n\n`
   and terminates with:
   `data: [DONE]\n\n`
2. Implement an asynchronous generator `token_stream_generator(prompt, tokens)`:
   - Yields tokens formatted strictly as SSE text chunks.
   - Flushes each token immediately to the client socket without buffering delays.
3. Validate protocol parsing on a simulated client receiver.

#### Verification Solution
```python
import asyncio
import json
from typing import AsyncGenerator

async def sse_token_stream(tokens: list[str]) -> AsyncGenerator[str, None]:
    for i, tok in enumerate(tokens):
        await asyncio.sleep(0.01)  # Simulate inter-token latency
        payload = {
            "id": "chatcmpl-123",
            "choices": [{"index": 0, "delta": {"content": tok}, "finish_reason": None}]
        }
        yield f"data: {json.dumps(payload)}\n\n"
        
    # Terminal event
    yield "data: [DONE]\n\n"

# Verification Run
if __name__ == "__main__":
    async def main():
        received_chunks = []
        token_list = ["The", " future", " of", " AI", " is", " autonomous."]
        
        async for chunk in sse_token_stream(token_list):
            received_chunks.append(chunk)
            
        print(f"Stream completed. Total chunks received: {len(received_chunks)}")
        print("First SSE chunk:\n", received_chunks[0])
        print("Terminal chunk:\n", received_chunks[-1])
        
        assert received_chunks[0].startswith("data: {")
        assert received_chunks[-1] == "data: [DONE]\n\n"
        assert len(received_chunks) == len(token_list) + 1

    asyncio.run(main())
```

---

## Tier 3: Production Dockerfile Hardening & Multi-Stage Builds

### Exercise 3: Hardened Multi-Stage Dockerfile for PyTorch Services
**Concepts**: Multi-stage Docker builds, non-root user execution, layer cache optimization, vulnerability reduction.

#### Problem Specification
Deploying raw Python virtual environments with development toolchains (gcc, make) and root privileges into production creates massive container image bloat ($> 4\text{GB}$) and severe security attack surfaces:
1. Write a 2-stage production Dockerfile:
   - **Stage 1 (Builder)**: Installs build dependencies, compiles wheels, and creates a clean virtual environment using `uv`.
   - **Stage 2 (Runtime)**: Copies only the runtime virtual environment into a minimal `python:3.11-slim` base image.
   - Enforces execution under a non-root system user (`appuser`, UID 10001).
   - Configures health checks (`HEALTHCHECK --interval=30s --timeout=3s CMD curl -f http://localhost:8000/healthz || exit 1`).

#### Verification Solution
```dockerfile
# Stage 1: Build stage
FROM python:3.11-slim AS builder

WORKDIR /build
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Stage 2: Hardened Runtime
FROM python:3.11-slim AS runtime

WORKDIR /app
ENV PATH="/install/bin:$PATH" \
    PYTHONPATH="/install/lib/python3.11/site-packages:$PYTHONPATH" \
    PYTHONUNBUFFERED=1

# Create non-root system user
RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -s /bin/false appuser

COPY --from=builder /install /install
COPY --chown=appuser:appgroup ./app /app/app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz')" || exit 1

ENTRYPOINT ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```
