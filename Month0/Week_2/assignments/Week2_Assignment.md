# Week 2 Production Engineering Assignment: Enterprise Resilient Async Orchestrator & Task Pipeline

## Executive Overview & Architectural Motivation
In production machine learning systems (such as inference gateways, embedding extractors, and multi-tenant feature stores), upstream endpoints frequently experience transient network partitions, downstream database backpressure, rate-limit throttling, and intermittent model inference timeouts. 

Naively issuing synchronous HTTP/RPC requests or blindly retrying requests without jittered exponential backoff leads to **thundering herd problems** and server cascading failures.

In this assignment, you will architect a production-grade, asynchronous, resilient micro-orchestration engine named `AsyncPipelineOrchestrator` featuring:
1. **Parameterized Jittered Exponential Backoff Decorator** (`@with_retry`).
2. **Token Bucket Rate Limiting Decorator** (`@rate_limited`).
3. **Structured Pydantic v2 / Invariant Validation Pipeline Contracts**.
4. **Cooperative Concurrency Task Scheduler with Structured Exception Aggregation** (`asyncio.TaskGroup` / `asyncio.gather` with cancellation boundaries).

---

## Technical Specifications & System Architecture

```
                    ┌───────────────────────────────────────────────┐
                    │          Client Batch Ingestion Request       │
                    └───────────────────────┬───────────────────────┘
                                            │
                                            ▼
                    ┌───────────────────────────────────────────────┐
                    │      Pydantic Task Payload Validation         │
                    └───────────────────────┬───────────────────────┘
                                            │
                                            ▼
                    ┌───────────────────────────────────────────────┐
                    │       AsyncPipelineOrchestrator Scheduler     │
                    │  (Bounded Concurrency Semaphore + TaskGroup) │
                    └───────┬───────────────────────────────┬───────┘
                            │                               │
             Worker Coroutine 1              Worker Coroutine 2
             ┌────────────────────────┐      ┌────────────────────────┐
             │ Token Bucket Rate Gate │      │ Token Bucket Rate Gate │
             └──────────────┬─────────┘      └──────────────┬─────────┘
                            │                               │
             ┌──────────────▼─────────┐      ┌──────────────▼─────────┐
             │ Jittered Retry Handler │      │ Jittered Retry Handler │
             └──────────────┬─────────┘      └──────────────┬─────────┘
                            │                               │
             ┌──────────────▼─────────┐      ┌──────────────▼─────────┐
             │ Upstream Model Inference│     │ Upstream Model Inference│
             └────────────────────────┘      └────────────────────────┘
```

---

## Detailed Deliverable Requirements

### 1. Jittered Exponential Backoff Decorator (`resilience.py`)
Implement `@with_retry`:
- **Formula**: The delay before retry $k$ (where $k \in [1, \text{max\_retries}]$) must follow:
  $$\text{delay}_k = \min\left(\text{max\_delay}, \text{base\_delay} \times 2^{k-1}\right) + \text{Uniform}(0, \text{jitter})$$
- Supports asynchronous coroutine functions (`async def`) transparently.
- Preserves full metadata (`@functools.wraps`) and parameter signatures.
- Must filter retries by a tuple of retriable exception classes (e.g., `(TimeoutError, ConnectionResetError)`), immediately bubbling non-retriable exceptions (e.g., `ValueError`, `KeyError`).

### 2. Async Token Bucket Rate Limiter (`limiter.py`)
Implement an asynchronous token bucket algorithm:
- Rate capacity $C$ tokens, replenishment rate $r$ tokens/sec.
- Asynchronous acquisition: `await limiter.acquire()` suspends the task non-blockingly until tokens are replenished.
- Thread-safe / Coroutine-safe state locking via `asyncio.Lock`.

### 3. Core Orchestration Engine (`orchestrator.py`)
Implement `AsyncPipelineOrchestrator`:
- Manages bounded concurrent execution with an `asyncio.Semaphore(concurrency_limit)`.
- Dispatches a batch of tasks $T_1, T_2, \dots, T_N$ concurrently.
- Implements graceful cancellation: if a fatal pipeline-level exception occurs, cancel inflight tasks and collect diagnostic traces.
- Emits structured telemetry summary: total runtime, successful tasks, failed tasks, and retry count distribution.

---

## Starter Code Template & Submission Skeleton

```python
"""
resilience_orchestrator.py - Production Async Pipeline
"""
import asyncio
import functools
import random
import time
from dataclasses import dataclass, field
from typing import Callable, Coroutine, Any, Type, Tuple, List, Optional

@dataclass
class ExecutionMetrics:
    total_tasks: int = 0
    successful_tasks: int = 0
    failed_tasks: int = 0
    total_retries: int = 0
    elapsed_seconds: float = 0.0

class TokenBucketLimiter:
    """Async Token Bucket Rate Limiter."""
    def __init__(self, rate_per_second: float, capacity: float):
        self.rate = rate_per_second
        self.capacity = capacity
        self.tokens = capacity
        self.last_updated = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: float = 1.0) -> None:
        async with self._lock:
            while True:
                now = time.monotonic()
                elapsed = now - self.last_updated
                self.last_updated = now
                self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
                
                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return
                
                # Sleep exact deficit duration
                deficit = tokens - self.tokens
                sleep_duration = deficit / self.rate
                await asyncio.sleep(sleep_duration)

def with_retry(
    max_retries: int = 3,
    base_delay: float = 0.1,
    max_delay: float = 2.0,
    jitter: float = 0.05,
    retriable_exceptions: Tuple[Type[BaseException], ...] = (Exception,)
):
    """Decorator applying exponential backoff with full jitter to async functions."""
    def decorator(func: Callable[..., Coroutine[Any, Any, Any]]):
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            attempt = 0
            while True:
                try:
                    return await func(*args, **kwargs)
                except retriable_exceptions as exc:
                    attempt += 1
                    if attempt > max_retries:
                        raise RuntimeError(f"Max retries ({max_retries}) exceeded.") from exc
                    
                    delay = min(max_delay, base_delay * (2 ** (attempt - 1))) + random.uniform(0, jitter)
                    await asyncio.sleep(delay)
        return wrapper
    return decorator

class AsyncPipelineOrchestrator:
    def __init__(self, concurrency_limit: int = 5, rate_limit_rps: float = 10.0):
        self.semaphore = asyncio.Semaphore(concurrency_limit)
        self.limiter = TokenBucketLimiter(rate_per_second=rate_limit_rps, capacity=rate_limit_rps)
        self.metrics = ExecutionMetrics()

    async def execute_task(self, task_id: int, payload: dict) -> dict:
        async with self.semaphore:
            await self.limiter.acquire()
            return await self._process_payload(task_id, payload)

    @with_retry(max_retries=3, base_delay=0.05, retriable_exceptions=(TimeoutError, ConnectionResetError))
    async def _process_payload(self, task_id: int, payload: dict) -> dict:
        # Simulated stochastic network call
        if random.random() < 0.3:
            raise ConnectionResetError(f"Flaky network dropped task {task_id}")
        await asyncio.sleep(0.02)
        return {"task_id": task_id, "status": "COMPLETED", "result": payload.get("data", 0) ** 2}

    async def run_batch(self, tasks: List[dict]) -> List[dict]:
        t0 = time.monotonic()
        self.metrics.total_tasks = len(tasks)
        
        async def safe_run(idx: int, task: dict):
            try:
                res = await self.execute_task(idx, task)
                self.metrics.successful_tasks += 1
                return res
            except Exception as e:
                self.metrics.failed_tasks += 1
                return {"task_id": idx, "status": "FAILED", "error": str(e)}

        coros = [safe_run(i, t) for i, t in enumerate(tasks)]
        results = await asyncio.gather(*coros)
        self.metrics.elapsed_seconds = time.monotonic() - t0
        return results

# Verification Harness
if __name__ == "__main__":
    async def main():
        orchestrator = AsyncPipelineOrchestrator(concurrency_limit=4, rate_limit_rps=20.0)
        sample_batch = [{"data": i} for i in range(20)]
        results = await orchestrator.run_batch(sample_batch)
        print(f"Batch completed in {orchestrator.metrics.elapsed_seconds:.3f}s")
        print(f"Successes: {orchestrator.metrics.successful_tasks} | Failures: {orchestrator.metrics.failed_tasks}")
        print("First 3 results:", results[:3])

    asyncio.run(main())
```

---

## Grading Rubric (100 Points Total)

| Component | Points | Criteria |
| :--- | :--- | :--- |
| **Retry & Jitter Mathematics** | 25 | Correct exponential backoff formula, bounded jitter, selective exception filtering, metadata preservation. |
| **Token Bucket Limiter** | 25 | Zero busy-waiting loops, non-blocking asynchronous suspension via deficit division, thread/coroutine lock protection. |
| **Bounded Concurrency & Scheduler** | 25 | Strict adherence to concurrency limits using `asyncio.Semaphore`, correct handling of task futures. |
| **Resilience & Graceful Degrade** | 15 | Robust exception isolation; one failing task does not contaminate or terminate sibling tasks. |
| **Telemetry & Verification Suite** | 10 | Complete metrics instrumentation and automated test assertions. |
