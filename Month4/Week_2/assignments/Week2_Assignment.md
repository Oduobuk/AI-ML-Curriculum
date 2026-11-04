# Week 2 Production Engineering Assignment: High-Throughput vLLM-Style Continuous Batching & PagedAttention Serving Engine

## Executive Overview & Architectural Motivation
Serving open-weights Large Language Models (Llama-3, Mistral, Qwen) in high-concurrency production environments requires overcoming GPU memory fragmentation and request head-of-line blocking. Traditional serving engines waste up to $80\%$ of GPU memory due to over-allocated contiguous KV-caches and fail to saturate Tensor Cores because of static batching delays.

In this assignment, you will architect a **High-Throughput In-Memory LLM Inference Engine** inspired by **vLLM**.

You will build a virtual memory **PagedAttention Block Table Manager**, an iteration-level **Continuous Batching Scheduler**, an activation-aware weight quantization (AWQ) simulator, and an asynchronous HTTP server endpoint exposing an OpenAI-compatible `/v1/completions` API.

---

## Technical Specifications & Architecture Diagram

```
                 Incoming Asynchronous Client Requests (/v1/completions)
                                            │
                                            ▼
                 ┌───────────────────────────────────────────────────────┐
                 │       Continuous Batching Request Scheduler           │
                 │   - Dynamic Admission Control                         │
                 │   - Iteration-Level Multiplexing (Decode Steps)       │
                 │   - Preempt / Evict Completed Requests                │
                 └──────────────────────────┬────────────────────────────┘
                                            │
                                            ▼
                 ┌───────────────────────────────────────────────────────┐
                 │       PagedAttention Virtual Memory Manager           │
                 │   - Non-contiguous Physical KV Blocks (BlockSize=16)  │
                 │   - Dynamic Block Allocation & Table Lookups          │
                 │   - Zero External Memory Fragmentation                │
                 └──────────────────────────┬────────────────────────────┘
                                            │
                                            ▼
                 ┌───────────────────────────────────────────────────────┐
                 │          Quantized Tensor Engine (AWQ INT4)           │
                 │   - Fused GEMM Matrix Vector Multiplication           │
                 │   - Sub-5ms Inter-Token Latency (ITL)                 │
                 └──────────────────────────┬────────────────────────────┘
                                            │
                                            ▼
                 Asynchronous Streaming Response Generator (SSE)
```

---

## Detailed Deliverable Requirements

### 1. PagedAttention Virtual Block Manager (`paged_cache.py`)
Implement `PagedKVCacheManager`:
- Allocates fixed-size tensor memory blocks on CPU/GPU (`block_size = 16`).
- Maps logical sequence positions to physical block IDs via a dynamic translation table.
- Implements copy-on-write branching for parallel sampling or prefix caching.
- Handles block release and reclamation without heap re-allocations.

### 2. Continuous Batching Iteration Scheduler (`scheduler.py`)
Implement `IterationScheduler`:
- Manages request state machines: `WAITING`, `RUNNING`, `FINISHED`.
- Multiplexes forward execution across requests at distinct generational phases (some in prefill, some in step 10 of decode).
- Computes aggregate serving metrics:
  - Throughput (tokens/sec).
  - Time-To-First-Token (TTFT).
  - Cache utilization percentage: $\frac{\text{Allocated Blocks}}{\text{Total Physical Blocks}}$.

### 3. Asynchronous HTTP Server (`serving_api.py`)
- Wrap the engine into a lightweight asynchronous FastAPI or pure `asyncio` server.
- Expose `/v1/completions` accepting JSON payloads:
  `{"prompt": "...", "max_tokens": 32, "temperature": 0.7}`.
- Stream tokens incrementally back to the client using Server-Sent Events (SSE).

---

## Starter Code Template & Submission Skeleton

```python
"""
vllm_continuous_batching_engine.py - Production LLM Serving Engine
"""
import asyncio
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

@dataclass
class ServingRequest:
    request_id: str
    prompt: str
    max_tokens: int
    tokens_generated: int = 0
    created_at: float = field(default_factory=time.time)
    first_token_time: Optional[float] = None
    finished: bool = False
    output_tokens: List[str] = field(default_factory=list)

class PhysicalBlockPool:
    def __init__(self, num_blocks: int, block_size: int = 16):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.free_block_ids: List[int] = list(range(num_blocks))

    def allocate(self) -> int:
        if not self.free_block_ids:
            raise MemoryError("Paged KV-Cache memory pool exhausted!")
        return self.free_block_ids.pop()

    def free(self, block_id: int):
        self.free_block_ids.append(block_id)

class ProductionServingScheduler:
    def __init__(self, max_batch_size: int = 4, total_blocks: int = 64):
        self.max_batch_size = max_batch_size
        self.block_pool = PhysicalBlockPool(total_blocks)
        self.request_block_tables: Dict[str, List[int]] = {}
        
        self.waiting_queue: List[ServingRequest] = []
        self.running_batch: List[ServingRequest] = []
        self.finished_requests: List[ServingRequest] = []

    def submit_request(self, req: ServingRequest):
        self.waiting_queue.append(req)

    def step(self):
        # 1. Admit waiting requests
        while len(self.running_batch) < self.max_batch_size and self.waiting_queue:
            req = self.waiting_queue.pop(0)
            initial_block = self.block_pool.allocate()
            self.request_block_tables[req.request_id] = [initial_block]
            self.running_batch.append(req)

        if not self.running_batch:
            return

        # 2. Simulate forward token decode iteration
        still_running = []
        for req in self.running_batch:
            if req.first_token_time is None:
                req.first_token_time = time.time()
                
            req.tokens_generated += 1
            req.output_tokens.append(f"tok_{req.tokens_generated}")
            
            # Check block allocation boundary
            if req.tokens_generated % self.block_pool.block_size == 0:
                new_block = self.block_pool.allocate()
                self.request_block_tables[req.request_id].append(new_block)

            # Check termination
            if req.tokens_generated >= req.max_tokens:
                req.finished = True
                self.finished_requests.append(req)
                # Free memory blocks
                for b_id in self.request_block_tables.pop(req.request_id, []):
                    self.block_pool.free(b_id)
            else:
                still_running.append(req)
                
        self.running_batch = still_running

# Verification Run
if __name__ == "__main__":
    scheduler = ProductionServingScheduler(max_batch_size=2, total_blocks=16)
    
    # Submit requests
    scheduler.submit_request(ServingRequest("REQ-1", "Explain Quantum Mechanics", max_tokens=10))
    scheduler.submit_request(ServingRequest("REQ-2", "Write a Python script", max_tokens=4))
    scheduler.submit_request(ServingRequest("REQ-3", "Summarize AI in 2026", max_tokens=6))
    
    iteration = 0
    while scheduler.running_batch or scheduler.waiting_queue:
        scheduler.step()
        iteration += 1
        
    print(f"Finished serving 3 concurrent requests across {iteration} iterations.")
    print("Free physical blocks remaining:", len(scheduler.block_pool.free_block_ids))
    assert len(scheduler.finished_requests) == 3
    assert len(scheduler.block_pool.free_block_ids) == 16
    print("All physical memory blocks cleanly reclaimed!")
```

---

## Grading Rubric (100 Points Total)

| Component | Points | Criteria |
| :--- | :--- | :--- |
| **PagedAttention Block Table Manager** | 35 | Flawless block allocation, logical-to-physical mapping, zero external fragmentation. |
| **Continuous Batching Scheduler** | 30 | Correct iteration-level step mechanics, instant request eviction and queue backfilling. |
| **Quantization & AWQ Understanding** | 15 | Accurate mathematical demonstration of channel scaling and low-bit truncation error. |
| **Asynchronous API & Streaming** | 10 | Clean non-blocking request handling with token streaming design. |
| **Verification & Memory Reclamation** | 10 | Complete automated test suite verifying $100\%$ block memory reclamation upon completion. |
