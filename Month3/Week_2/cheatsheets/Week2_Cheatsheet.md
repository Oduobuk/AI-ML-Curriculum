# Month 3, Week 2 Cheatsheet: Neural Network Engineering, Decoupled AdamW & Normalization

---

## 1. Weight Initialization & Activation Variance Preservation

To avoid vanishing or exploding activations across deep layers:

### Linear / Tanh (Xavier / Glorot)
$$\text{Var}(W) = \frac{2}{n_{\text{in}} + n_{\text{out}}} \qquad W_{i, j} \sim \mathcal{N}\left(0, \frac{2}{n_{\text{in}} + n_{\text{out}}}\right)$$

### ReLU / GeLU (He / Kaiming)
$$\text{Var}(W) = \frac{2}{n_{\text{in}}} \qquad W_{i, j} \sim \mathcal{N}\left(0, \frac{2}{n_{\text{in}}}\right)$$

```python
import torch.nn as nn

nn.init.kaiming_normal_(linear_layer.weight, nonlinearity="relu")
nn.init.zeros_(linear_layer.bias)
```

---

## 2. Pre-Norm vs Post-Norm Stability in Deep Residual Networks

```
            Post-LayerNorm (Unstable at Depth > 12)
   x ──► [Residual Function F(x)] ──► (+) ──► [LayerNorm] ──► Output
   │                                   ▲
   └───────────────────────────────────┘

            Pre-LayerNorm / Pre-RMSNorm (Stable at Depth 100+)
   x ──► [RMSNorm] ──► [Residual Function F(x)] ──► (+) ──► Output
   │                                                 ▲
   └─────────────────────────────────────────────────┘
```

---

## 3. RMSNorm Architecture & Mathematical Formulation

Root Mean Square Normalization dispenses with mean centering, reducing FLOPs by $\sim 15\%$ while preserving scale invariance:
$$\text{RMSNorm}(x) = \frac{x}{\text{RMS}(x)} \odot \gamma \quad \text{where } \text{RMS}(x) = \sqrt{\frac{1}{d} \sum_{i=1}^d x_i^2 + \epsilon}$$

```python
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight
```

---

## 4. Decoupled Weight Decay in AdamW (Loshchilov & Hutter 2017)

In standard Adam with $L_2$ decay, gradients are penalized before moment estimation:
$$\tilde{g}_t = g_t + \lambda \theta_{t-1} \implies \text{Second moment } v_t \text{ accumulates } \lambda^2 \theta^2 \text{ (distorting updates!)}$$

In **AdamW**, weight decay is applied directly to the parameters, completely decoupled from gradient moments:
$$\theta_t = \theta_{t-1} - \eta \lambda \theta_{t-1} - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

---

## 5. Gradient Norm Clipping

Prevents exploding gradient spikes during stochastic mini-batch training:
$$\text{If } \|\mathbf{g}\|_2 > \text{max\_norm}, \quad \mathbf{g} \leftarrow \mathbf{g} \times \frac{\text{max\_norm}}{\|\mathbf{g}\|_2}$$

```python
import torch.nn.utils as utils

loss.backward()
total_norm = utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```
