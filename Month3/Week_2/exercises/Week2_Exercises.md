# Week 2: Graduate-Level Problem Set — Neural Network Engineering, Decoupled AdamW & LayerNorm/RMSNorm

Welcome to the Month 3 Week 2 Problem Set. This suite rigorous covers the mathematical engineering foundations of modern neural architectures: Xavier/Kaiming weight initialization, gradient clipping dynamics, Pre-LayerNorm vs Post-LayerNorm stability, RMSNorm simplification, and decoupled weight decay mechanics in AdamW.

---

## Tier 1: Core Mechanics & Parameter Initialization

### Exercise 1: Kaiming (He) vs Xavier (Glorot) Initialization Variance Preservation
**Concepts**: Activation variance forward/backward propagation, $\text{Var}(y) = n_{\text{in}} \text{Var}(w) \text{Var}(x)$, ReLU dead neurons.

#### Problem Specification
Consider a linear layer $y = W x$ where $x \in \mathbb{R}^{d_{\text{in}}}$ has independent zero-mean components with variance $\sigma_x^2$:
1. Prove that for linear/tanh activations, maintaining $\text{Var}(y_i) = \text{Var}(x_j)$ requires:
   $$\text{Var}(W_{i, j}) = \frac{1}{d_{\text{in}}}$$
2. For ReLU non-linearities ($\text{ReLU}(z) = \max(0, z)$), exactly half of the activations are zeroed out ($\mathbb{E}[\text{ReLU}(z)^2] = \frac{1}{2} \text{Var}(z)$). Prove that preserving variance requires:
   $$\text{Var}(W_{i, j}) = \frac{2}{d_{\text{in}}} \implies W_{i, j} \sim \mathcal{N}\left(0, \frac{2}{d_{\text{in}}}\right)$$
3. Write a simulation initializing a 20-layer MLP with:
   - Standard Normal $\mathcal{N}(0, 1)$
   - Xavier Normal $\mathcal{N}(0, \frac{1}{d})$
   - Kaiming Normal $\mathcal{N}(0, \frac{2}{d})$
4. Measure activation variance at layer 20. Demonstrate how standard initialization vanishes or explodes activations.

#### Verification Solution
```python
import torch
import torch.nn as nn
import math

def simulate_initialization_variance():
    torch.manual_seed(42)
    d = 256
    depth = 20
    x = torch.randn(1000, d)
    
    # 1. Kaiming
    x_kaiming = x.clone()
    for _ in range(depth):
        w = torch.randn(d, d) * math.sqrt(2.0 / d)
        x_kaiming = torch.relu(x_kaiming @ w.t())
    var_kaiming = x_kaiming.var().item()
    
    # 2. Naive Standard Normal
    x_naive = x.clone()
    for _ in range(depth):
        w = torch.randn(d, d)
        x_naive = torch.relu(x_naive @ w.t())
    var_naive = x_naive.var().item()
    
    print(f"Initial Input Variance:      {x.var().item():.4f}")
    print(f"Layer 20 Variance (Kaiming): {var_kaiming:.4f}")
    print(f"Layer 20 Variance (Naive):   {var_naive:.4e} (Exploded/Vanished)")
    
    assert 0.1 < var_kaiming < 5.0
    assert var_naive > 1e10 or math.isnan(var_naive)

if __name__ == "__main__":
    simulate_initialization_variance()
```

---

## Tier 2: Normalization Dynamics: LayerNorm vs RMSNorm

### Exercise 2: Pure PyTorch RMSNorm Implementation & Backward Derivative
**Concepts**: Root Mean Square Normalization (RMSNorm), scale invariance, inference latency.

#### Problem Specification
Modern LLMs (Llama-3, Mistral, Gemma) replace LayerNorm with RMSNorm, discarding mean centering to reduce kernel FLOPs:
$$\text{RMSNorm}(x) = \frac{x}{\text{RMS}(x)} \odot \gamma \quad \text{where } \text{RMS}(x) = \sqrt{\frac{1}{d} \sum_{i=1}^d x_i^2 + \epsilon}$$
1. Implement `RMSNorm(nn.Module)` in pure PyTorch without using `torch.nn.LayerNorm`.
2. Compare the runtime latency and memory footprint between `nn.LayerNorm` and `RMSNorm` over 1,000 forward passes of shape `(32, 2048, 4096)`.
3. Assert that both modules produce identical scale-invariant gradient dynamics.

#### Verification Solution
```python
import torch
import torch.nn as nn
import time

class PureRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RMS = sqrt(mean(x^2) + eps)
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.gamma

# Verification Run
if __name__ == "__main__":
    torch.manual_seed(42)
    B, S, D = 8, 128, 512
    x = torch.randn(B, S, D)
    
    norm = PureRMSNorm(D)
    out = norm(x)
    
    # Check that root mean square of out is approximately 1.0 (since gamma is ones)
    computed_rms = torch.sqrt(torch.mean(out ** 2, dim=-1)).mean().item()
    print("Computed Output RMS:", computed_rms)
    assert abs(computed_rms - 1.0) < 0.01
```

---

## Tier 3: Optimization Dynamics: Decoupled Weight Decay in AdamW

### Exercise 3: AdamW vs L2-Regularized Adam on Adaptive Moments
**Concepts**: Decoupled weight decay (Loshchilov & Hutter 2017), gradient distortion, moment scaling.

#### Problem Specification
In standard Adam with L2 regularization, the weight penalty $\lambda \theta$ is added directly to the gradient:
$$\tilde{g}_t = g_t + \lambda \theta_{t-1}$$
This distorts the second-moment accumulator $v_t \leftarrow \beta_2 v_{t-1} + (1-\beta_2)\tilde{g}_t^2$, suppressing updates for large weights rather than decaying them uniformly.
1. Implement `DecoupledAdamW(torch.optim.Optimizer)` from scratch.
2. Verify that parameters with identical weight magnitudes decay at the exact rate $\theta_t \leftarrow (1 - \eta \lambda) \theta_{t-1}$ regardless of their gradient frequency.

#### Verification Solution
```python
import torch
from torch.optim import Optimizer

class CustomAdamW(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            wd = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1
                step = state['step']

                # 1. First decoupled weight decay update
                p.mul_(1.0 - lr * wd)

                # 2. Update biased first and second moments
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                # 3. Bias correction
                bias_correction1 = 1.0 - beta1 ** step
                bias_correction2 = 1.0 - beta2 ** step
                step_size = lr / bias_correction1

                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss

# Verification Run
if __name__ == "__main__":
    w = torch.tensor([5.0, -3.0], requires_grad=True)
    opt = CustomAdamW([w], lr=0.1, weight_decay=0.05)
    
    # Run step with zero gradient -> pure weight decay
    loss = (w * 0.0).sum()
    loss.backward()
    opt.step()
    
    print("Parameter after decoupled step with 0 grad:", w.data)
    # Should decay by exactly (1 - lr * wd) = 1 - 0.1 * 0.05 = 0.995
    assert torch.isclose(w.data[0], torch.tensor(5.0 * 0.995))
```
