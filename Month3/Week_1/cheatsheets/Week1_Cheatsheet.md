# Month 3, Week 1 Cheatsheet: PyTorch 2.x Internals, Tensor Strides & Custom Autograd VJP

---

## 1. PyTorch Tensor Storage & Memory Stride Mechanics

In PyTorch, a `torch.Tensor` is a strided window over a 1D contiguous block of memory (`untyped_storage`):

```
                   Physical 1D Storage Buffer in RAM / VRAM
┌───────┬───────┬───────┬───────┬───────┬───────┬───────┬───────┬───────┬───────┬───────┬───────┐
│   0   │   1   │   2   │   3   │   4   │   5   │   6   │   7   │   8   │   9   │  10   │  11   │
└───────┴───────┴───────┴───────┴───────┴───────┴───────┴───────┴───────┴───────┴───────┴───────┘
  ▲                               ▲                               ▲
  │                               │                               │
  Row 0 (offset 0)                Row 1 (offset 4)                Row 2 (offset 8)
  Stride: (4, 1) -> To advance 1 row, jump 4 elements; to advance 1 column, jump 1 element.
```

### Stride & Contiguity Rules
```python
import torch

A = torch.randn(3, 4)
A.stride()           # (4, 1)
A.is_contiguous()    # True

# Transpose modifies metadata strides without copying data
B = A.t()
B.stride()           # (1, 4)
B.is_contiguous()    # False

# B.view(12) fails because elements are non-contiguous in memory
B_cont = B.contiguous()  # Allocates new memory buffer with stride (3, 1)
B_cont.view(12)          # Succeeded!
```

---

## 2. Custom Autograd Vector-Jacobian Product (VJP)

To implement custom forward/backward logic without computational graph overhead:
```python
import torch
import math

class CustomGeLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x)
        c = math.sqrt(2.0 / math.pi)
        inner = c * (x + 0.044715 * torch.pow(x, 3))
        tanh_out = torch.tanh(inner)
        ctx.c = c
        return 0.5 * x * (1.0 + tanh_out)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        (x,) = ctx.saved_tensors
        c = ctx.c
        inner = c * (x + 0.044715 * torch.pow(x, 3))
        tanh_out = torch.tanh(inner)
        sech2 = 1.0 - torch.pow(tanh_out, 2)
        du_dx = c * (1.0 + 3.0 * 0.044715 * torch.pow(x, 2))
        dx = 0.5 * (1.0 + tanh_out) + 0.5 * x * sech2 * du_dx
        return grad_output * dx
```

---

## 3. Finite Difference Gradient Verification (`gradcheck`)

Always verify analytical backward VJP kernels using double precision:
```python
# gradcheck requires float64 and requires_grad=True
test_tensor = torch.randn(4, 4, dtype=torch.float64, requires_grad=True)
passed = torch.autograd.gradcheck(CustomGeLUFunction.apply, (test_tensor,), eps=1e-6, atol=1e-4)
assert passed  # Raises an error if numerical and analytical gradients diverge!
```

---

## 4. PyTorch 2.x `torch.compile` & Inductor Fusion

PyTorch 2.x captures graphs via TorchDynamo and generates fused C++/Triton kernels via TorchInductor:
```python
import torch

model = MyDeepModel().to("cuda")
# Fuses sequential elementwise kernels (Linear + Bias + GELU) into single GPU kernel launches
compiled_model = torch.compile(model, mode="reduce-overhead")
```
