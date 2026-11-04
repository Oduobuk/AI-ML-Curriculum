# Week 1: Graduate-Level Problem Set — PyTorch 2.x Internals, Tensor Strides & Custom Autograd VJP

Welcome to the Month 3 Week 1 Problem Set. This suite evaluates your mastery of modern deep learning tensor runtimes: PyTorch 2.x computational graphs, tensor memory stride layouts (contiguous vs strided), custom backward Vector-Jacobian Product (VJP) kernels via `torch.autograd.Function`, and analytical numerical gradient checks.

---

## Tier 1: Tensor Memory Layout & Stride Mechanics

### Exercise 1: Tensor Strides, Views, and Memory Contiguity
**Concepts**: `tensor.stride()`, `tensor.storage()`, `tensor.is_contiguous()`, zero-copy reshaping vs memory copying.

#### Problem Specification
In PyTorch, tensors are strided views over a flat 1D memory storage buffer:
1. Create a 2D tensor $A \in \mathbb{R}^{3 \times 4}$.
2. Inspect its strides: `A.stride()`.
3. Compute the transpose $B = A^T$.
   - Does $B$ allocate new physical memory storage? Inspect `B.storage().data_ptr()` vs `A.storage().data_ptr()`.
   - What are the strides of $B$? Is $B$ contiguous (`B.is_contiguous()`)?
4. Call `B.view(12)`. Explain why this raises a `RuntimeError`, and write the correct operation (`B.contiguous().view(12)`).

#### Verification Solution
```python
import torch

def inspect_tensor_strides():
    A = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    print("Tensor A shape:", A.shape, "| Strides:", A.stride())
    assert A.stride() == (4, 1)
    assert A.is_contiguous()
    
    # Transpose
    B = A.t()
    print("Tensor B (A.T) shape:", B.shape, "| Strides:", B.stride())
    assert B.stride() == (1, 4)
    assert not B.is_contiguous()
    assert A.untyped_storage().data_ptr() == B.untyped_storage().data_ptr()
    
    # Verify non-contiguous view failure
    try:
        _ = B.view(12)
        assert False, "Should raise RuntimeError"
    except RuntimeError as e:
        print("Successfully caught expected non-contiguous view error:", e)
        
    B_cont = B.contiguous()
    assert B_cont.is_contiguous()
    assert B_cont.view(12).shape == (12,)
    print("Contiguous view operation succeeded.")

if __name__ == "__main__":
    inspect_tensor_strides()
```

---

## Tier 2: Custom Autograd Vector-Jacobian Product (VJP)

### Exercise 2: Custom Autograd Layer for GeLU Activation
**Concepts**: `torch.autograd.Function`, forward evaluation, backward gradient propagation, chain rule.

#### Problem Specification
The Gaussian Error Linear Unit (GeLU) is the standard activation function for modern Transformers (BERT, GPT, ViT):
$$\text{GeLU}(x) = x \cdot \Phi(x) \approx 0.5 x \left(1 + \tanh\left(\sqrt{\frac{2}{\pi}} \left(x + 0.044715 x^3\right)\right)\right)$$
1. Implement a custom `torch.autograd.Function` named `CustomGeLUFunction`.
2. In `forward(ctx, x)`: save necessary tensors using `ctx.save_for_backward()`.
3. In `backward(ctx, grad_output)`: derive and implement the exact analytical derivative $\frac{d}{dx} \text{GeLU}(x) \cdot \text{grad\_output}$.
4. Verify your analytical gradient using `torch.autograd.gradcheck` with `eps=1e-6` and `atol=1e-4`.

#### Verification Solution
```python
import torch
import math

class CustomGeLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x)
        # Approximate GeLU
        c = math.sqrt(2.0 / math.pi)
        inner = c * (x + 0.044715 * torch.pow(x, 3))
        tanh_out = torch.tanh(inner)
        out = 0.5 * x * (1.0 + tanh_out)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        (x,) = ctx.saved_tensors
        c = math.sqrt(2.0 / math.pi)
        inner = c * (x + 0.044715 * torch.pow(x, 3))
        tanh_out = torch.tanh(inner)
        sech2 = 1.0 - torch.pow(tanh_out, 2)
        
        # d/dx [ 0.5 * x * (1 + tanh(u)) ] = 0.5 * (1 + tanh(u)) + 0.5 * x * sech^2(u) * du/dx
        du_dx = c * (1.0 + 3.0 * 0.044715 * torch.pow(x, 2))
        dx = 0.5 * (1.0 + tanh_out) + 0.5 * x * sech2 * du_dx
        return grad_output * dx

# Verification Run
if __name__ == "__main__":
    torch.manual_seed(42)
    # Double precision required for gradcheck
    test_input = torch.randn(4, 4, dtype=torch.float64, requires_grad=True)
    
    # Run gradcheck
    passed = torch.autograd.gradcheck(CustomGeLUFunction.apply, (test_input,), eps=1e-6, atol=1e-4)
    print(f"torch.autograd.gradcheck passed: {passed}")
    assert passed
```

---

## Tier 3: Memory Efficient Scaled Dot-Product Attention from Scratch

### Exercise 3: Numerically Stable Softmax Attention & Backward VJP
**Concepts**: FlashAttention principles, online softmax normalization, memory-efficient attention.

#### Problem Specification
1. Standard Scaled Dot-Product Attention computes:
   $$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$
2. When sequence length $S$ is large, the $S \times S$ intermediate attention matrix exhausts GPU VRAM ($O(S^2)$ memory).
3. Implement a custom autograd Function `ScaledDotProductAttentionFunction`:
   - Subtract row-maximum $\max_j (S_{i, j})$ before exponentiation to eliminate numerical overflow.
   - Analytically compute the backward pass gradients $\nabla_Q, \nabla_K, \nabla_V$.
4. Benchmark memory footprint against native PyTorch.

#### Verification Solution
```python
import torch
import math

class CustomAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        scale = 1.0 / math.sqrt(Q.shape[-1])
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
        
        # Stable softmax
        scores_max = torch.max(scores, dim=-1, keepdim=True).values
        exp_scores = torch.exp(scores - scores_max)
        probs = exp_scores / torch.sum(exp_scores, dim=-1, keepdim=True)
        
        out = torch.matmul(probs, V)
        ctx.save_for_backward(Q, K, V, probs)
        ctx.scale = scale
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        Q, K, V, probs = ctx.saved_tensors
        scale = ctx.scale
        
        # dL/dV = P^T @ grad_out
        grad_V = torch.matmul(probs.transpose(-2, -1), grad_out)
        
        # dL/dP = grad_out @ V^T
        grad_P = torch.matmul(grad_out, V.transpose(-2, -1))
        
        # Backward through Softmax: dL/dS = P * (grad_P - sum(grad_P * P, dim=-1, keepdim=True))
        sum_grad_P = torch.sum(grad_P * probs, dim=-1, keepdim=True)
        grad_S = probs * (grad_P - sum_grad_P) * scale
        
        # dL/dQ = grad_S @ K
        grad_Q = torch.matmul(grad_S, K)
        # dL/dK = grad_S^T @ Q
        grad_K = torch.matmul(grad_S.transpose(-2, -1), Q)
        
        return grad_Q, grad_K, grad_V

# Verification Run
if __name__ == "__main__":
    torch.manual_seed(42)
    B, H, S, D = 2, 4, 8, 16
    Q = torch.randn(B, H, S, D, dtype=torch.float64, requires_grad=True)
    K = torch.randn(B, H, S, D, dtype=torch.float64, requires_grad=True)
    V = torch.randn(B, H, S, D, dtype=torch.float64, requires_grad=True)
    
    passed = torch.autograd.gradcheck(CustomAttentionFunction.apply, (Q, K, V), eps=1e-5, atol=1e-4)
    print(f"Custom Attention gradcheck passed: {passed}")
    assert passed
```
