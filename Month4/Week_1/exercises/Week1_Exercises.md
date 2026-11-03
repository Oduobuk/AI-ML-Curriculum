# Week 1: Graduate-Level Problem Set — Transformer Architecture Decoded, KV-Caching & Rotary Position Embeddings (RoPE)

Welcome to the Month 4 Week 1 Problem Set. This suite evaluates your mastery of the modern autoregressive Transformer decoder (Llama-3, Mistral, Gemma): Rotary Position Embeddings (RoPE) complex vector rotations, causal triangular autoregressive masking, Grouped-Query Attention (GQA) memory footprints, and stateful Key-Value (KV) cache allocation profiling.

---

## Tier 1: Core Mechanics & Rotary Position Embeddings (RoPE)

### Exercise 1: Complex 2D Rotary Position Embedding (RoPE) Kernel
**Concepts**: Su et al. (2021) RoPE formulation, rotation matrices $R_{\Theta, m}^d$, inner product relative preservation.

#### Problem Specification
Standard absolute positional embeddings fail to generalize across extended context lengths. RoPE encodes relative token distance by rotating 2D query and key slices in the complex plane:
1. For query vector coordinate pair $(q_{2i}, q_{2i+1})$ at sequence position $m$:
   $$\mathbf{R}_{\theta_i, m} \begin{pmatrix} q_{2i} \\ q_{2i+1} \end{pmatrix} = \begin{pmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{pmatrix} \begin{pmatrix} q_{2i} \\ q_{2i+1} \end{pmatrix}$$
   where $\theta_i = \theta_{\text{base}}^{-2i / d}$ and $\theta_{\text{base}} = 10,000$ (or $500,000$ in Llama-3).
2. Prove that the inner product between rotated query at position $m$ and key at position $n$ depends strictly on relative offset $m - n$:
   $$\langle \mathbf{R}_m q, \mathbf{R}_n k \rangle = \text{Re}\left[ (q \odot k^*) e^{i(m - n)\theta} \right]$$
3. Implement `apply_rotary_emb(x, freqs_cos, freqs_sin)` using pure PyTorch tensor operations.

#### Verification Solution
```python
import torch

def precompute_rope_frequencies(dim: int, max_seq_len: int, theta_base: float = 10000.0) -> tuple[torch.Tensor, torch.Tensor]:
    # dim must be even
    theta = 1.0 / (theta_base ** (torch.arange(0, dim, 2).float() / dim))
    m = torch.arange(max_seq_len).float()
    freqs = torch.outer(m, theta)  # [max_seq_len x dim/2]
    
    freqs_cos = torch.cos(freqs)
    freqs_sin = torch.sin(freqs)
    return freqs_cos, freqs_sin

def apply_rotary_emb(x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor) -> torch.Tensor:
    # x: [B, Seq_Len, Heads, Head_Dim]
    seq_len = x.shape[1]
    cos = freqs_cos[:seq_len].unsqueeze(0).unsqueeze(2)  # [1, Seq_Len, 1, Head_Dim/2]
    sin = freqs_sin[:seq_len].unsqueeze(0).unsqueeze(2)
    
    # Reshape x into pairs: [B, Seq_Len, Heads, Head_Dim/2, 2]
    x_pairs = x.reshape(*x.shape[:-1], -1, 2)
    x0, x1 = x_pairs[..., 0], x_pairs[..., 1]
    
    # Apply rotation: [x0*cos - x1*sin, x0*sin + x1*cos]
    out0 = x0 * cos - x1 * sin
    out1 = x0 * sin + x1 * cos
    
    return torch.stack([out0, out1], dim=-1).flatten(-2)

# Verification Run
if __name__ == "__main__":
    torch.manual_seed(42)
    B, Seq, Heads, Dim = 2, 8, 4, 32
    cos, sin = precompute_rope_frequencies(Dim, 128)
    q = torch.randn(B, Seq, Heads, Dim)
    
    q_rot = apply_rotary_emb(q, cos, sin)
    print("Rotated Query Tensor Shape:", q_rot.shape)
    assert q_rot.shape == q.shape
    
    # Verify rotation preserves norm
    orig_norm = torch.norm(q, dim=-1)
    rot_norm = torch.norm(q_rot, dim=-1)
    assert torch.allclose(orig_norm, rot_norm, atol=1e-5)
    print("RoPE norm invariance verified successfully!")
```

---

## Tier 2: Grouped-Query Attention (GQA) & KV Cache Optimization

### Exercise 2: Grouped-Query Attention (GQA) Memory Footprint & Tensor Broadcasting
**Concepts**: Multi-Head Attention (MHA), Multi-Query Attention (MQA), Grouped-Query Attention (GQA), KV cache memory bandwidth.

#### Problem Specification
In Multi-Head Attention ($H_Q = 32$ heads), storing distinct KV heads for long contexts saturates GPU high-bandwidth memory (HBM). GQA pools queries into $G$ groups, sharing each Key/Value head across $\frac{H_Q}{H_{KV}}$ query heads:
1. Let context length $S = 8192$, layers $L = 32$, head dimension $D_h = 128$, float16 precision (2 bytes).
2. Calculate the total KV cache memory required per concurrent batch request:
   - For standard MHA ($H_Q = 32, H_{KV} = 32$):
     $$\text{Mem}_{\text{MHA}} = 2 \times L \times 2 \times B \times S \times H_{KV} \times D_h \times 2\text{ bytes}$$
   - For Llama-3 style GQA ($H_Q = 32, H_{KV} = 8$):
     $$\text{Mem}_{\text{GQA}} = \frac{8}{32} \text{Mem}_{\text{MHA}} = 0.25 \times \text{Mem}_{\text{MHA}}$$
3. Implement `repeat_kv(x, n_rep)` in PyTorch to expand shared Key/Value tensors via zero-copy expansion.

#### Verification Solution
```python
import torch

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    x: [B, Seq_Len, Num_KV_Heads, Head_Dim]
    Expands Num_KV_Heads to (Num_KV_Heads * n_rep) without allocating memory.
    """
    if n_rep == 1:
        return x
    B, S, n_kv_heads, D = x.shape
    return x[:, :, :, None, :].expand(B, S, n_kv_heads, n_rep, D).reshape(B, S, n_kv_heads * n_rep, D)

# Verification Run
if __name__ == "__main__":
    B, S, H_kv, D = 2, 16, 4, 64
    kv_tensor = torch.randn(B, S, H_kv, D)
    expanded = repeat_kv(kv_tensor, n_rep=4)
    
    print(f"Original KV Shape: {kv_tensor.shape}")
    print(f"Expanded KV Shape: {expanded.shape} (Expected 16 query heads)")
    assert expanded.shape == (B, S, 16, D)
```

---

## Tier 3: Stateful Autoregressive Decoding Loop with KV-Cache

### Exercise 3: Prefill vs Autoregressive Token Generation with Stateful KV-Cache
**Concepts**: Time-To-First-Token (TTFT), Inter-Token Latency (ITL), cached attention complexity reduction from $O(S^2)$ to $O(S)$.

#### Problem Specification
1. Implement a complete stateful KV-Cache manager `KVCache`:
   - Stores pre-allocated `k_cache` and `v_cache` tensors of shape `(B, Max_Seq_Len, H_kv, D_h)`.
   - Supports `update(k_new, v_new, start_pos)`.
2. Demonstrate that during generation, step $t$ computes attention only between the current single query token ($1 \times D$) and historical keys ($t \times D$), avoiding $O(t^2)$ recomputation.

#### Verification Solution
```python
import torch
import math

class StaticKVCache:
    def __init__(self, batch_size: int, max_seq_len: int, num_kv_heads: int, head_dim: int):
        self.k_cache = torch.zeros(batch_size, max_seq_len, num_kv_heads, head_dim)
        self.v_cache = torch.zeros(batch_size, max_seq_len, num_kv_heads, head_dim)

    def update(self, start_pos: int, k_val: torch.Tensor, v_val: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = k_val.shape[1]
        self.k_cache[:, start_pos:start_pos + seq_len] = k_val
        self.v_cache[:, start_pos:start_pos + seq_len] = v_val
        return self.k_cache[:, :start_pos + seq_len], self.v_cache[:, :start_pos + seq_len]

# Verification Run
if __name__ == "__main__":
    cache = StaticKVCache(batch_size=1, max_seq_len=64, num_kv_heads=2, head_dim=16)
    
    # 1. Prefill step (5 tokens)
    k_prefill = torch.randn(1, 5, 2, 16)
    v_prefill = torch.randn(1, 5, 2, 16)
    k_all, v_all = cache.update(0, k_prefill, v_prefill)
    assert k_all.shape == (1, 5, 2, 16)
    
    # 2. Decode step (1 token at pos 5)
    k_decode = torch.randn(1, 1, 2, 16)
    v_decode = torch.randn(1, 1, 2, 16)
    k_all, v_all = cache.update(5, k_decode, v_decode)
    assert k_all.shape == (1, 6, 2, 16)
    print("KV Cache update verified: active sequence length extended to 6.")
```
