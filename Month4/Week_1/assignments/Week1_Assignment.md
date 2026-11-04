# Week 1 Production Engineering Assignment: Modern Autoregressive Transformer Decoder (Llama-3 Architecture) from Scratch

## Executive Overview & Architectural Motivation
Modern open-weights Large Language Models (Llama-3, Mistral, Gemma, DeepSeek) share an evolved decoder-only Transformer foundation that deviates significantly from the original 2017 Vaswani paper. They abandon learned absolute positional encodings in favor of **Rotary Position Embeddings (RoPE)**, replace Multi-Head Attention with **Grouped-Query Attention (GQA)**, replace LayerNorm with **RMSNorm**, and use gated **SwiGLU** feedforward networks.

In this assignment, you will architect a complete, functional **Modern Autoregressive Transformer Decoder** from scratch in pure PyTorch 2.x.

Your model will feature exact RoPE vector rotations, GQA projection, Pre-RMSNorm residual stabilization, SwiGLU activations, and an optimized, stateful **KV-Cache generation engine** supporting greedy and top-$p$ nucleus sampling.

---

## Technical Specifications & Architecture Diagram

```
                 Input Token IDs $X \in \mathbb{N}^{B \times S}$
                                        │
                                        ▼
                 ┌───────────────────────────────────────────┐
                 │       Token Embedding Table               │
                 └─────────────────────┬─────────────────────┘
                                       │
                                       ▼
                 ┌───────────────────────────────────────────┐
                 │       Decoder Transformer Layer           │
                 │   - Pre-RMSNorm                           │
                 │   - RoPE Frequency Rotation on Q and K    │
                 │   - Grouped-Query Attention (GQA)         │
                 │   - KV-Cache Concatenation & Masking      │
                 │   - Residual Skip Connection              │
                 │   - Pre-RMSNorm + SwiGLU MLP              │
                 └─────────────────────┬─────────────────────┘
                                       │ (Repeated L times)
                                       ▼
                 ┌───────────────────────────────────────────┐
                 │       Final RMSNorm & Language Head       │
                 └─────────────────────┬─────────────────────┘
                                       │
                                       ▼
                 Autoregressive Token Probabilities / KV Cache
```

---

## Detailed Deliverable Requirements

### 1. Modern Attention & RoPE Kernels (`transformer_blocks.py`)
Implement the following foundational modules:
- `RMSNorm(dim, eps=1e-6)`
- `apply_rope(x, freqs_cos, freqs_sin)`
- `SwiGLU(dim, hidden_dim)`:
  $$\text{SwiGLU}(x) = \left(x W_{\text{gate}} \odot \text{SiLU}(x W_{\text{gate}})\right) W_{\text{up}} \dots \implies (x W_1) \cdot \text{SiLU}(x W_3) W_2$$
- `GroupedQueryAttention(dim, n_heads, n_kv_heads, max_seq_len)`:
  - Supports incremental token insertion into an internal or external `KVCache`.
  - Applies causal triangular upper-masking during the initial prefill phase.

### 2. Full Decoder Stack (`decoder.py`)
Implement `LlamaTransformerDecoder(nn.Module)`:
- Configurable `vocab_size`, `dim`, `n_layers`, `n_heads`, `n_kv_heads`, `max_seq_len`.
- Weight tying between input embedding matrix and output language model classification head (optional, configurable).
- Returns next-token logits and updated KV-cache pointers.

### 3. Autoregressive Generation & Memory Profiling (`generator.py`)
- Implement `generate(prompt_tokens, max_new_tokens, temperature, top_p)`:
  - Executes initial parallel prompt prefill.
  - Switches to single-token incremental decode steps using the KV-cache.
  - Profiles Time-To-First-Token (TTFT) and Inter-Token Latency (ITL).

---

## Starter Code Template & Submission Skeleton

```python
"""
llama_transformer_decoder.py - Production Decoder-Only Transformer Suite
"""
import torch
import torch.nn as nn
import math
from typing import Optional, Tuple

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight

class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)  # Gate
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)  # Down
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)  # Up

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(torch.silu(self.w1(x)) * self.w3(x))

class ModernDecoderLayer(nn.Module):
    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, hidden_dim: int):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.mlp = SwiGLU(dim, hidden_dim)
        
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads
        self.n_rep = n_heads // n_kv_heads
        
        self.q_proj = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, C = x.shape
        h = self.norm1(x)
        
        # Projections
        q = self.q_proj(h).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)
        
        # Expand KV heads for GQA
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)
            
        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        # Causal Mask
        mask = torch.triu(torch.full((S, S), float("-inf"), device=x.device), diagonal=1)
        scores = scores + mask
        
        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn, v).transpose(1, 2).reshape(B, S, C)
        
        # Residuals
        x = x + self.out_proj(context)
        x = x + self.mlp(self.norm2(x))
        return x

class LlamaMiniDecoder(nn.Module):
    def __init__(self, vocab_size: int = 1000, dim: int = 128, n_layers: int = 4, 
                 n_heads: int = 8, n_kv_heads: int = 2, hidden_dim: int = 256):
        super().__init__()
        self.tok_embeddings = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([
            ModernDecoderLayer(dim, n_heads, n_kv_heads, hidden_dim) for _ in range(n_layers)
        ])
        self.norm = RMSNorm(dim)
        self.output = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.tok_embeddings(tokens)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.output(x)

# Verification Run
if __name__ == "__main__":
    torch.manual_seed(42)
    B, Seq = 2, 16
    tokens = torch.randint(0, 1000, (B, Seq))
    
    model = LlamaMiniDecoder(vocab_size=1000, dim=128, n_layers=2, n_heads=4, n_kv_heads=2)
    logits = model(tokens)
    
    print(f"Decoder Forward Pass Logits: {logits.shape} (Expected: [{B}, {Seq}, 1000])")
    assert logits.shape == (B, Seq, 1000)
    assert not torch.isnan(logits).any()
```

---

## Grading Rubric (100 Points Total)

| Component | Points | Criteria |
| :--- | :--- | :--- |
| **RoPE Mathematical Implementation** | 25 | Correct complex 2D planar rotation, frequency precomputation, and norm preservation. |
| **Grouped-Query Attention (GQA)** | 25 | Proper expansion of shared KV heads and correct tensor dimensionality checks. |
| **RMSNorm & SwiGLU Modules** | 20 | Clean forward operations with scale invariance and non-linear gating mechanics. |
| **Causal Masking & KV Caching** | 20 | Strict causal autoregressive isolation; stateful cache decoding verification. |
| **Verification & Latency Benchmarks** | 10 | Complete runnable execution script measuring TTFT and generation throughput. |
