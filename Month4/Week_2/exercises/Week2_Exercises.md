# Week 2: Graduate-Level Problem Set — Open LLMs, Weight Quantization & vLLM Serving Architecture

Welcome to the Month 4 Week 2 Problem Set. This suite evaluates your mastery of enterprise open-weights LLM deployment: Post-Training Quantization (PTQ) mechanics (AWQ, GPTQ, INT4/INT8), continuous batching request scheduling, virtual memory PagedAttention block managers, and vLLM serving optimization.

---

## Tier 1: Core Mechanics & Quantization Mathematics

### Exercise 1: Activation-Aware Weight Quantization (AWQ) Mathematical Simulation
**Concepts**: Weight clipping, salient channel preservation, activation magnitude scaling $W' = W \cdot S, X' = X \cdot S^{-1}$.

#### Problem Specification
Standard round-to-nearest (RTN) quantization damages LLM performance because a tiny fraction ($0.1\%$) of hidden feature channels exhibit huge activation outliers:
1. In AWQ (Lin et al. 2023), salient channels are protected by multiplying weight columns by a per-channel scale factor $s > 1$ before uniform INT4 quantization, and dividing input activations by $s$:
   $$W \cdot X = (W \cdot \text{diag}(s)) \cdot (\text{diag}(s)^{-1} X)$$
2. Let $s = s_X^\alpha$, where $s_X = \max |X|$ across the batch and $\alpha \in [0, 1]$ is a grid-search parameter.
3. Implement `awq_channel_scale(W, X, alpha=0.5)` in pure PyTorch.
4. Show that scaling down salient weight channels reduces overall mean squared error after INT4 truncation:
   $$\text{MSE} = \|W X - \text{dequantize}(\text{quantize}(W \cdot s)) \cdot s^{-1} X\|_F^2$$

#### Verification Solution
```python
import torch

def quantize_int4(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # Symmetric 4-bit quantization: [-8, 7]
    scale = torch.max(torch.abs(x), dim=-1, keepdim=True).values / 7.0
    scale = torch.clamp(scale, min=1e-5)
    q = torch.clamp(torch.round(x / scale), -8, 7)
    return q, scale

def dequantize_int4(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return q * scale

def evaluate_awq():
    torch.manual_seed(42)
    d_in, d_out = 64, 64
    W = torch.randn(d_out, d_in)
    
    # Activations with 2 extreme outlier channels
    X = torch.randn(10, d_in)
    X[:, 5] *= 30.0  # Outlier channel 5
    X[:, 20] *= 25.0 # Outlier channel 20
    
    Y_true = X @ W.t()
    
    # 1. Naive RTN Quantization
    q_naive, s_naive = quantize_int4(W)
    W_naive_dequant = dequantize_int4(q_naive, s_naive)
    Y_naive = X @ W_naive_dequant.t()
    mse_naive = torch.mean((Y_true - Y_naive) ** 2).item()
    
    # 2. AWQ Scale
    act_max = torch.max(torch.abs(X), dim=0).values
    s = (act_max ** 0.5).clamp(min=1e-2)
    s = s / s.mean()  # Normalize
    
    W_scaled = W * s.unsqueeze(0)
    q_awq, s_awq = quantize_int4(W_scaled)
    W_awq_dequant = dequantize_int4(q_awq, s_awq) / s.unsqueeze(0)
    Y_awq = X @ W_awq_dequant.t()
    mse_awq = torch.mean((Y_true - Y_awq) ** 2).item()
    
    print(f"Naive RTN INT4 MSE: {mse_naive:.4f}")
    print(f"AWQ-Scaled INT4 MSE: {mse_awq:.4f}")
    assert mse_awq < mse_naive
    print("AWQ demonstrated superior activation preservation under INT4 quantization!")

if __name__ == "__main__":
    evaluate_awq()
```

---

## Tier 2: PagedAttention Virtual Block Table Management

### Exercise 2: Virtual Memory PagedAttention Block Allocator
**Concepts**: Virtual-to-physical address translation, block size $B=16$, internal/external fragmentation elimination.

#### Problem Specification
Traditional LLM serving pre-allocates contiguous memory for maximum sequence length ($S=8192$), causing $60-80\%$ internal memory waste. PagedAttention divides the KV cache into fixed-size physical blocks (e.g. 16 tokens) mapped through a block table:
1. Implement `PagedBlockManager(num_physical_blocks=100, block_size=16)`:
   - Tracks free block IDs in a stack or list.
   - `allocate(request_id)`: assigns physical block IDs to a request's logical page table.
   - `append_token(request_id)`: writes a token into the current block; allocates a new physical block only when the current block is completely full.
   - `free(request_id)`: returns all associated blocks back to the free pool.
2. Demonstrate zero external memory fragmentation across 10 dynamically generating requests.

#### Verification Solution
```python
from typing import Dict, List

class PagedBlockManager:
    def __init__(self, num_physical_blocks: int = 64, block_size: int = 16):
        self.num_physical_blocks = num_physical_blocks
        self.block_size = block_size
        self.free_blocks: List[int] = list(range(num_physical_blocks))
        self.block_tables: Dict[str, List[int]] = {}
        self.token_counts: Dict[str, int] = {}

    def allocate_request(self, req_id: str) -> None:
        if not self.free_blocks:
            raise MemoryError("Out of physical KV blocks!")
        first_block = self.free_blocks.pop()
        self.block_tables[req_id] = [first_block]
        self.token_counts[req_id] = 0

    def append_token(self, req_id: str) -> None:
        count = self.token_counts[req_id]
        if count > 0 and count % self.block_size == 0:
            if not self.free_blocks:
                raise MemoryError("KV Cache full: cannot allocate new block for generation")
            new_block = self.free_blocks.pop()
            self.block_tables[req_id].append(new_block)
        self.token_counts[req_id] += 1

    def free_request(self, req_id: str) -> None:
        for block in self.block_tables.get(req_id, []):
            self.free_blocks.append(block)
        self.block_tables.pop(req_id, None)
        self.token_counts.pop(req_id, None)

# Verification Run
if __name__ == "__main__":
    mgr = PagedBlockManager(num_physical_blocks=10, block_size=16)
    mgr.allocate_request("req_01")
    assert len(mgr.block_tables["req_01"]) == 1
    
    # Append 32 tokens (spans exactly 2 blocks)
    for _ in range(32):
        mgr.append_token("req_01")
        
    print("Blocks allocated for req_01 (32 tokens):", mgr.block_tables["req_01"])
    assert len(mgr.block_tables["req_01"]) == 2
    assert len(mgr.free_blocks) == 8
    
    mgr.free_request("req_01")
    assert len(mgr.free_blocks) == 10
    print("PagedAttention block recycling verified successfully!")
```

---

## Tier 3: Continuous Batching Scheduling Engine

### Exercise 3: Iteration-Level Continuous Batching Engine
**Concepts**: Prefill vs decode iteration multiplexing, request queueing, dynamic batching.

#### Problem Specification
Static batching waits for the slowest generation in a batch to finish before accepting new prompts. Continuous batching (Orca/vLLM) schedules generation at the iteration level:
1. Implement `ContinuousBatchScheduler(max_batch_size=4)`:
   - Maintains `waiting_queue`, `running_batch`, and `finished_requests`.
   - In each tick: runs one forward decode step across all running requests.
   - Detects completion (`<EOS>` token or `max_tokens` reached) and immediately evacuates finished requests, backfilling from the `waiting_queue`.
2. Benchmark overall throughput (tokens per second) against a static batching baseline.

#### Verification Solution
```python
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class LLMRequest:
    req_id: str
    prompt_len: int
    max_tokens: int
    generated_tokens: int = 0
    is_finished: bool = False

class ContinuousBatchScheduler:
    def __init__(self, max_batch_size: int = 3):
        self.max_batch_size = max_batch_size
        self.waiting: List[LLMRequest] = []
        self.running: List[LLMRequest] = []
        self.completed: List[LLMRequest] = []
        self.total_step_count: int = 0

    def add_request(self, req: LLMRequest):
        self.waiting.append(req)

    def step(self):
        self.total_step_count += 1
        
        # 1. Admit waiting requests if capacity permits
        while len(self.running) < self.max_batch_size and self.waiting:
            admitted = self.waiting.pop(0)
            self.running.append(admitted)

        # 2. Simulate single-token decoding for running requests
        still_running = []
        for req in self.running:
            req.generated_tokens += 1
            if req.generated_tokens >= req.max_tokens:
                req.is_finished = True
                self.completed.append(req)
            else:
                still_running.append(req)
                
        self.running = still_running

# Verification Run
if __name__ == "__main__":
    scheduler = ContinuousBatchScheduler(max_batch_size=2)
    # 3 requests with variable length
    scheduler.add_request(LLMRequest("R1", prompt_len=4, max_tokens=2))
    scheduler.add_request(LLMRequest("R2", prompt_len=8, max_tokens=6))
    scheduler.add_request(LLMRequest("R3", prompt_len=2, max_tokens=3))
    
    steps = 0
    while scheduler.running or scheduler.waiting:
        scheduler.step()
        steps += 1
        
    print(f"Continuous batching finished all requests in {steps} iterations.")
    print("Completed Requests:", [r.req_id for r in scheduler.completed])
    assert len(scheduler.completed) == 3
```
