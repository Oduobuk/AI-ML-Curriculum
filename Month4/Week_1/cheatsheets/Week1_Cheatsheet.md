# Month 4, Week 1 Cheatsheet: Modern Decoder-Only Transformers, RoPE & KV-Caching

---

## 1. Modern Decoder Architecture Evolution (Vaswani 2017 vs Llama-3 2024+)

| Architectural Component | Original Transformer (2017) | Modern Open LLM (Llama-3 / Gemma) | Motivation & Benefit |
|:---|:---|:---|:---|
| **Positional Encoding** | Absolute Sinusoidal Table | Rotary Position Embedding (RoPE) | Relative token distance generalization across 128k+ contexts |
| **Attention Pattern** | Multi-Head Attention (MHA) | Grouped-Query Attention (GQA) | $4-8\times$ reduction in KV cache memory footprint |
| **Normalization** | Post-LayerNorm | Pre-RMSNorm | Eliminates activation explosion; saves $15\%$ compute |
| **Feedforward Layer** | Standard ReLU / GELU | Gated SwiGLU | Superior empirical language modeling density |

---

## 2. Rotary Position Embedding (RoPE) Mathematical Formulation

Encodes position by rotating 2D query/key coordinate pairs in the complex plane:
$$\mathbf{R}_{\Theta, m}^d = \text{diag}\left( R_{\theta_1, m}, R_{\theta_2, m}, \dots, R_{\theta_{d/2}, m} \right)$$
where each $2 \times 2$ rotation block is:
$$R_{\theta_i, m} = \begin{pmatrix} \cos(m \theta_i) & -\sin(m \theta_i) \\ \sin(m \theta_i) & \cos(m \theta_i) \end{pmatrix}, \qquad \theta_i = \theta_{\text{base}}^{-2(i-1)/d}$$
- **Inner Product Property**:
  $$\langle \mathbf{R}_m q, \mathbf{R}_n k \rangle = \text{Re}[(q \odot k^*) e^{i(m - n)\theta}]$$
  Depends strictly on relative position $m - n$, not absolute coordinates.

---

## 3. Grouped-Query Attention (GQA) & Memory Calculation

Queries have $H_Q$ heads; Keys and Values have $H_{KV}$ heads where $H_{KV} \ll H_Q$:
$$\text{Query-to-KV Head Ratio: } \frac{H_Q}{H_{KV}}$$

### KV-Cache Memory Formula
For batch size $B$, sequence length $S$, layers $L$, heads $H_{KV}$, head dimension $D_h$, and 16-bit precision (2 bytes):
$$\text{Memory}_{\text{KV}} = 2 \times L \times B \times S \times H_{KV} \times D_h \times 2 \text{ bytes}$$

---

## 4. Stateful KV-Cache Generation Engine

```python
class StaticKVCache:
    def __init__(self, batch_size: int, max_seq_len: int, num_kv_heads: int, head_dim: int):
        self.k_cache = torch.zeros(batch_size, max_seq_len, num_kv_heads, head_dim)
        self.v_cache = torch.zeros(batch_size, max_seq_len, num_kv_heads, head_dim)

    def update(self, pos: int, k_val: torch.Tensor, v_val: torch.Tensor):
        seq_len = k_val.shape[1]
        self.k_cache[:, pos:pos+seq_len] = k_val
        self.v_cache[:, pos:pos+seq_len] = v_val
        return self.k_cache[:, :pos+seq_len], self.v_cache[:, :pos+seq_len]
```
- **Prefill Phase**: Computes prompt tokens in parallel with causal upper-triangular mask ($O(S^2)$).
- **Decode Phase**: Generates 1 token per step using cached history ($O(S)$ per token).
