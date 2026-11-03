# Week 1 Production Engineering Assignment: Custom PyTorch 2.x Autograd Neural Engine with Stride Optimization

## Executive Overview & Architectural Motivation
To build production-grade foundational models, understand custom GPU acceleration kernels (Triton/CUDA), and debug training instability, AI engineers must master the low-level internals of deep learning runtimes: memory stride geometry, dynamic autograd computation graphs, and custom backward Vector-Jacobian Product (VJP) kernels.

In this assignment, you will engineer a **Custom Autograd Neural Layer & Profiling Suite** using modern **PyTorch 2.x**.

You will build custom `torch.autograd.Function` implementations for fused linear transformations and non-linear activations, execute analytical Jacobian verification via `gradcheck`, profile zero-copy tensor slicing across strided dimensions, and benchmark memory allocation and peak FLOP throughput against native PyTorch layers.

---

## Technical Specifications & Architecture Diagram

```
                 Input Activation Tensor $X \in \mathbb{R}^{B \times D_{in}}$
                                        │
                                        ▼
                 ┌───────────────────────────────────────────┐
                 │       Fused Linear + GeLU Custom VJP      │
                 │   - Forward: Saves inputs for backward    │
                 │   - Backward: Computes dX, dW, db in C-   │
                 │     contiguous memory layouts             │
                 └─────────────────────┬─────────────────────┘
                                       │
                                       ▼
                 ┌───────────────────────────────────────────┐
                 │       Autograd Graph Verification         │
                 │   - Numerical Finite Difference Check     │
                 │   - torch.autograd.gradcheck(atol=1e-4)   │
                 └─────────────────────┬─────────────────────┘
                                       │
                                       ▼
                 ┌───────────────────────────────────────────┐
                 │       PyTorch Profiler & Benchmark        │
                 │   - cuda/cpu memory footprint analysis    │
                 │   - FLOP estimation & Operator Fusion     │
                 └───────────────────────────────────────────┘
```

---

## Detailed Deliverable Requirements

### 1. Custom Fused Linear Layer with Backward VJP (`layers.py`)
Implement `FusedLinearFunction(torch.autograd.Function)`:
- **Forward Pass**: $Y = X W^T + b$ followed by approximate GeLU.
- **Backward Pass**:
  - Gradient w.r.t bias $b$: $\nabla_b = \sum_{\text{batch}} \delta$.
  - Gradient w.r.t weights $W$: $\nabla_W = \delta^T X$.
  - Gradient w.r.t input $X$: $\nabla_X = \delta W$.
  where $\delta = \text{grad\_output} \odot \text{GeLU}'(X W^T + b)$.
- Wrap the function into a standard `torch.nn.Module` named `FusedLinearGeLU`.

### 2. Numerical Gradient Parity Verification (`gradcheck.py`)
- Execute double-precision finite difference checks across varying batch sizes ($B \in [2, 8]$) and hidden dimensions ($D \in [4, 16]$).
- Enforce strict `torch.autograd.gradcheck` assertions.

### 3. Profiler Benchmark & Memory Stride Audit (`profiler.py`)
- Benchmark `FusedLinearGeLU` against standard sequential `nn.Sequential(nn.Linear, nn.GELU)`:
  - Track peak memory allocation using `torch.cuda.max_memory_allocated()` or CPU `tracemalloc`.
  - Profile kernel call overhead and operator graph depth using `torch.profiler.profile()`.

---

## Starter Code Template & Submission Skeleton

```python
"""
custom_autograd_engine.py - Production Custom Autograd Suite
"""
import torch
import torch.nn as nn
import math
import time

class FusedLinearGeLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X: torch.Tensor, W: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # Pre-activation
        Z = torch.matmul(X, W.t()) + b
        
        # GeLU forward approximation
        c = math.sqrt(2.0 / math.pi)
        inner = c * (Z + 0.044715 * torch.pow(Z, 3))
        tanh_out = torch.tanh(inner)
        Y = 0.5 * Z * (1.0 + tanh_out)
        
        ctx.save_for_backward(X, W, b, Z, tanh_out)
        ctx.c = c
        return Y

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        X, W, b, Z, tanh_out = ctx.saved_tensors
        c = ctx.c
        
        # Derivative of GeLU w.r.t Z
        sech2 = 1.0 - torch.pow(tanh_out, 2)
        du_dz = c * (1.0 + 3.0 * 0.044715 * torch.pow(Z, 2))
        dgelu_dz = 0.5 * (1.0 + tanh_out) + 0.5 * Z * sech2 * du_dz
        
        delta = grad_output * dgelu_dz  # [B x D_out]
        
        # Gradients
        grad_X = torch.matmul(delta, W)                      # [B x D_in]
        grad_W = torch.matmul(delta.t(), X)                  # [D_out x D_in]
        grad_b = torch.sum(delta, dim=0)                     # [D_out]
        
        return grad_X, grad_W, grad_b

class FusedLinearGeLU(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.W = nn.Parameter(torch.randn(out_features, in_features) / math.sqrt(in_features))
        self.b = nn.Parameter(torch.zeros(out_features))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return FusedLinearGeLUFunction.apply(X, self.W, self.b)

# Verification Run
if __name__ == "__main__":
    torch.manual_seed(42)
    B, D_in, D_out = 4, 8, 6
    
    # 1. Gradcheck
    X = torch.randn(B, D_in, dtype=torch.float64, requires_grad=True)
    W = torch.randn(D_out, D_in, dtype=torch.float64, requires_grad=True)
    b = torch.randn(D_out, dtype=torch.float64, requires_grad=True)
    
    passed = torch.autograd.gradcheck(FusedLinearGeLUFunction.apply, (X, W, b), eps=1e-6, atol=1e-4)
    print(f"FusedLinearGeLU gradcheck passed: {passed}")
    assert passed
    
    # 2. Module Forward & Backward Test
    layer = FusedLinearGeLU(16, 32)
    inputs = torch.randn(64, 16)
    out = layer(inputs)
    loss = out.sum()
    loss.backward()
    
    print(f"Layer forward output shape: {out.shape}")
    print(f"Weight gradient norm: {layer.W.grad.norm().item():.4f}")
    assert layer.W.grad is not None
```

---

## Grading Rubric (100 Points Total)

| Component | Points | Criteria |
| :--- | :--- | :--- |
| **Mathematical Derivation & VJP Correctness** | 35 | Flawless chain-rule derivation of $\nabla_X, \nabla_W, \nabla_b$ through GeLU. |
| **Numerical Gradient Verification** | 25 | Passing `torch.autograd.gradcheck` with zero assertion failures. |
| **Tensor Stride & Memory Architecture** | 20 | Explicit handling of contiguous memory buffers and stride safety. |
| **PyTorch Module Integration** | 10 | Clean encapsulation into an idiomatic, parameter-registered `nn.Module`. |
| **Profiling & Benchmark Metrics** | 10 | Memory footprint and runtime execution benchmarks against standard PyTorch. |
