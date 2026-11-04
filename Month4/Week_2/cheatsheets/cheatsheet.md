# Month 4, Week 2 Cheatsheet: Open LLMs, Weight Quantization & vLLM Serving Architecture

---

## 1. Post-Training Quantization (PTQ) Paradigms: RTN vs GPTQ vs AWQ

| Quantization Method | Granularity | Outlier Handling Strategy | Perplexity Degradation (4-bit) |
|:---|:---|:---|:---|
| **Round-To-Nearest (RTN)** | Uniform / Group | None (clips outliers) | Catastrophic on $< 13\text{B}$ models |
| **GPTQ (Frantar et al.)** | Layer-wise | Second-order inverse Hessian compensation | Minimal; requires calibration data |
| **AWQ (Lin et al.)** | Per-channel scale | Scales salient activation channels $W' = W \cdot S$ | Near-lossless; preserves salient weights |

### Activation-Aware Weight Quantization (AWQ) Core Principle
A tiny fraction ($0.1\%$) of hidden channels carry salient feature information. Instead of keeping them in FP16, AWQ scales them:
$$W \cdot X = (W \cdot \text{diag}(s)) \cdot (\text{diag}(s)^{-1} X)$$
- $s = s_X^\alpha$ where $s_X = \max |X|$ across the batch.
- Protects salient weights from INT4 rounding truncation without hardware-unfriendly mixed precision.

---

## 2. Virtual Memory PagedAttention (vLLM Architecture)

Eliminates memory waste from dynamic sequence length allocation:

```
               Logical Token Sequence vs Physical Memory Blocks
 ┌────────────────────────────────────────────────────────┐
 │ Logical Tokens: [t0, t1, t2 ... t15] [t16 ... t31]     │
 └──────────────────────────┬─────────────────────────────┘
                            │ Virtual-to-Physical Block Table
                            ▼
 ┌───────────────────┬───────────────────┬────────────────┐
 │ Physical Block 7  │ Physical Block 2  │ Free Block 12  │
 │ (Tokens 0-15)     │ (Tokens 16-31)    │ (Unallocated)  │
 └───────────────────┴───────────────────┴────────────────┘
 (Zero external fragmentation; dynamically allocates physical blocks on demand)
```

---

## 3. Iteration-Level Continuous Batching (Orca / vLLM)

Static batching blocks an entire batch until the longest request completes generation. Continuous batching operates at the iteration level:
- New requests enter the batch at the very next decoding step.
- Finished requests leave immediately upon generating `<EOS>` or reaching `max_tokens`.
- Saturates GPU compute utilization ($> 85\%$).

---

## 4. Serving Performance Latency Metrics

- **Time-To-First-Token (TTFT)**: Latency from HTTP request arrival to the emission of the first streamed token (governed by prefill FLOPs).
- **Inter-Token Latency (ITL)**: Time between successive tokens during generation (governed by memory bandwidth).
- **Throughput**: Global tokens generated per second across all concurrent client streams.
