# Week 1: Graduate-Level Problem Set — Fine-Tuning, Parameter-Efficient Fine-Tuning (PEFT) & QLoRA

Welcome to the Month 5 Week 1 Problem Set. This suite evaluates your mastery of enterprise post-training optimization: Parameter-Efficient Fine-Tuning (PEFT), Low-Rank Adaptation (LoRA) matrix decomposition, NormalFloat-4 (NF4) theoretical information density, double dequantization, and fused weight merging for zero-overhead inference.

---

## Tier 1: Core Mechanics & Low-Rank Adaptation (LoRA)

### Exercise 1: LoRA Forward Pass, Scaling Factor & Parameter Count
**Concepts**: Intrinsic rank hypothesis (Aghajanyan et al. 2020), low-rank factorization $W = W_0 + \frac{\alpha}{r} B A$, rank $r \ll d$.

#### Problem Specification
Standard full-parameter fine-tuning of an LLM linear layer $W_0 \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ requires updating $d_{\text{in}} \times d_{\text{out}}$ weights and allocating $2 \times$ optimizer states in Adam (FP32). LoRA freezes $W_0$ and injects trainable rank-$r$ adapters:
$$h = W_0 x + \frac{\alpha}{r} B A x$$
where $A \in \mathbb{R}^{r \times d_{\text{in}}}$ is initialized from $\mathcal{N}(0, \sigma^2)$ and $B \in \mathbb{R}^{d_{\text{out}} \times r}$ is initialized to exact zero.
1. Let $d_{\text{in}} = 4096, d_{\text{out}} = 4096, r = 16$.
   - Calculate total trainable parameters for full fine-tuning vs LoRA.
   - What is the compression ratio?
2. Why is $B$ initialized to zero? What would happen during the first forward pass if both $A$ and $B$ were initialized with random standard normal noise?
3. Implement `PurePyTorchLoRALinear(in_features, out_features, rank=16, alpha=32.0)` from scratch.

#### Verification Solution
```python
import torch
import torch.nn as nn
import math

class PurePyTorchLoRALinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int = 16, alpha: float = 32.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scaling = alpha / rank

        # Frozen base weight
        self.base_weight = nn.Parameter(torch.randn(out_features, in_features) / math.sqrt(in_features), requires_grad=False)
        
        # Trainable low-rank matrices
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) / math.sqrt(in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))  # Must be zero initialized

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base forward pass
        base_out = x @ self.base_weight.t()
        # LoRA adaptation: (x @ A^T) @ B^T * scaling
        lora_out = (x @ self.lora_A.t()) @ self.lora_B.t() * self.scaling
        return base_out + lora_out

    def merge_weights(self) -> torch.Tensor:
        """Merges adapter into base weight for zero-overhead inference."""
        return self.base_weight + (self.lora_B @ self.lora_A) * self.scaling

# Verification Run
if __name__ == "__main__":
    torch.manual_seed(42)
    B, D_in, D_out, R = 4, 128, 128, 8
    x = torch.randn(B, D_in)
    
    layer = PurePyTorchLoRALinear(D_in, D_out, rank=R, alpha=16.0)
    
    # Initial forward pass: lora_B is zeros, so output must strictly match base weight
    initial_out = layer(x)
    base_expected = x @ layer.base_weight.t()
    assert torch.allclose(initial_out, base_expected, atol=1e-6)
    print("Zero-initialization check passed: LoRA perturbation is exactly 0 on step 0.")
    
    # Train step simulation
    loss = initial_out.sum()
    loss.backward()
    assert layer.lora_B.grad is not None
    assert layer.base_weight.grad is None  # Base weight remained frozen
    print("Parameter freezing verified: Base weight gradients are None.")
```

---

## Tier 2: NormalFloat-4 (NF4) Theoretical Information Density

### Exercise 2: Construction of the 16-Level NormalFloat-4 (NF4) Codebook
**Concepts**: Information-theoretic quantization, quantiles of $\mathcal{N}(0, 1)$, Dettmers et al. (2023) QLoRA.

#### Problem Specification
Standard integer INT4 uniform quantization assumes a uniform distribution of values. Neural network weights follow a normal distribution $\mathcal{N}(0, \sigma^2)$, meaning uniform bins waste representation bits on the sparse tails.
1. The 16 quantiles $q_i$ of the NormalFloat-4 (NF4) codebook are derived such that each bin has equal probability mass $\frac{1}{2k}$ under $\mathcal{N}(0, 1)$:
   $$q_i = \frac{1}{2} \left( Q_X\left(\frac{i}{2k}\right) + Q_X\left(\frac{i+1}{2k}\right) \right)$$
   where $Q_X$ is the quantile function (inverse CDF) of $\mathcal{N}(0, 1)$, normalized so that the interval is bounded in $[-1, 1]$.
2. Compute the exact 16-point NF4 codebook using `scipy.stats.norm.ppf`.
3. Quantize a normal tensor into NF4 indices and calculate the Signal-to-Quantization-Noise Ratio (SQNR) compared to uniform INT4.

#### Verification Solution
```python
import numpy as np
from scipy.stats import norm

def generate_nf4_codebook() -> np.ndarray:
    # 16 points for 4-bit representation
    v1 = norm.ppf(np.linspace(0.5 / 16, 15.5 / 16, 16))
    # Symmetrize and normalize to [-1, 1]
    v1 = v1 / np.max(np.abs(v1))
    return v1

def quantize_to_nf4(weights: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    # Scale to [-1, 1]
    abs_max = np.max(np.abs(weights))
    norm_w = weights / abs_max
    
    # Nearest neighbor assignment in codebook
    diffs = np.abs(norm_w[..., None] - codebook)
    indices = np.argmin(diffs, axis=-1)
    
    # Dequantize
    dequantized = codebook[indices] * abs_max
    return dequantized

# Verification Run
if __name__ == "__main__":
    np.random.seed(42)
    codebook = generate_nf4_codebook()
    print("NF4 16-Point Normalized Codebook:")
    print(np.round(codebook, 4))
    
    weights = np.random.normal(loc=0.0, scale=0.02, size=10_000)
    dequant = quantize_to_nf4(weights, codebook)
    
    sqnr = 10 * np.log10(np.sum(weights ** 2) / np.sum((weights - dequant) ** 2))
    print(f"NF4 Reconstruction SQNR: {sqnr:.2f} dB")
    assert sqnr > 20.0
```

---

## Tier 3: Zero-Overhead LoRA Weight Merging & Export

### Exercise 3: Weight Merging and De-Merging for High-Throughput Serving
**Concepts**: Weight folding, inference latency elimination, multi-adapter hot-swapping.

#### Problem Specification
During training, low-rank adapters are computed as separate matrix branches. In production inference, computing $W_0 x + \frac{\alpha}{r} B A x$ incurs additional GEMM kernel launches and memory reads.
1. Implement `merge()`: $W_{\text{prod}} = W_0 + \frac{\alpha}{r} (B A)$.
2. Implement `unmerge()`: $W_0 = W_{\text{prod}} - \frac{\alpha}{r} (B A)$.
3. Prove that after merging, inference forward latency matches standard linear layers without any adapter penalty.

#### Verification Solution
```python
import torch
import torch.nn as nn

class MergeableLoRALinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.scaling = alpha / rank
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.lora_A = nn.Parameter(torch.randn(rank, in_features))
        self.lora_B = nn.Parameter(torch.randn(out_features, rank))
        self.merged = False

    def merge(self):
        if not self.merged:
            with torch.no_grad():
                self.weight.data += (self.lora_B @ self.lora_A) * self.scaling
            self.merged = True

    def unmerge(self):
        if self.merged:
            with torch.no_grad():
                self.weight.data -= (self.lora_B @ self.lora_A) * self.scaling
            self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.merged:
            return x @ self.weight.t()
        return (x @ self.weight.t()) + (x @ self.lora_A.t()) @ self.lora_B.t() * self.scaling

# Verification Run
if __name__ == "__main__":
    layer = MergeableLoRALinear(32, 32, rank=4)
    x = torch.randn(1, 32)
    
    out_unmerged = layer(x)
    layer.merge()
    out_merged = layer(x)
    
    assert torch.allclose(out_unmerged, out_merged, atol=1e-5)
    print("Merged and unmerged outputs are mathematically identical!")
```
