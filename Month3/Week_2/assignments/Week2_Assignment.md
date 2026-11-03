# Week 2 Production Engineering Assignment: Pre-RMSNorm Deep Residual Classifier with Decoupled AdamW

## Executive Overview & Architectural Motivation
Deep feedforward networks without proper residual connections and normalization suffer from catastrophic gradient explosion or attenuation as depth increases beyond 10 layers. Furthermore, coupling L2 weight decay to gradient updates distorts adaptive learning rates in Adam.

In this assignment, you will engineer a **Deep Residual Tabular Classifier (ResNet-MLP)** featuring:
1. **Pre-RMSNorm Residual Blocks** with Kaiming initialization and dropout.
2. **From-Scratch Decoupled AdamW Optimizer** with cosine annealing learning rate scheduling.
3. **Automated Gradient Norm Clipping** to bound exploding gradients.
4. **PyTorch Mixed Precision (AMP `torch.autocast`) Training Harness**.

---

## Technical Specifications & Architecture Diagram

```
                 Input Features $x \in \mathbb{R}^{B \times D_{in}}$
                                   │
                                   ▼
                 ┌───────────────────────────────────┐
                 │       Initial Linear Projection   │
                 └─────────────────┬─────────────────┘
                                   │
                                   ▼
                 ┌───────────────────────────────────┐
                 │     Pre-RMSNorm Residual Block    │
                 │   x_norm = RMSNorm(x)             │
                 │   h = Linear(SwiGLU(Linear(x_norm)))│
                 │   out = x + Dropout(h)            │
                 └─────────────────┬─────────────────┘
                                   │ (Repeated N times)
                                   ▼
                 ┌───────────────────────────────────┐
                 │       Final RMSNorm & Head        │
                 └─────────────────┬─────────────────┘
                                   │
                                   ▼
                 Calibrated Class Logits & Decoupled Weight Decay
```

---

## Detailed Deliverable Requirements

### 1. Pre-RMSNorm Residual Block (`blocks.py`)
Implement `PreRMSNormResBlock(nn.Module)`:
- Normalizes input activations using `PureRMSNorm` before the feedforward layers.
- Uses SwiGLU or GELU activation in the hidden layer.
- Adds the residual identity skip connection: $x_{l+1} = x_l + \text{Dropout}(\mathcal{F}(\text{RMSNorm}(x_l)))$.
- Initializes weights using Kaiming normal with gain $\sqrt{2.0}$.

### 2. Custom Decoupled AdamW Optimizer (`optimizer.py`)
- Implement `DecoupledAdamW` adhering to the PyTorch `Optimizer` API.
- Implements direct parameter shrinkage: $\theta_t \leftarrow (1 - \eta \lambda) \theta_{t-1}$.
- Includes unbiased first and second moment estimators.

### 3. Mixed Precision Training & Gradient Clipping Loop (`train.py`)
- Train a 12-layer deep ResNet-MLP on a challenging high-dimensional classification dataset.
- Implement gradient norm clipping via `torch.nn.utils.clip_grad_norm_(max_norm=1.0)`.
- Use `torch.amp.autocast("cpu")` or `torch.amp.autocast("cuda")` to execute forward passes in lower precision (`bfloat16`/`float16`).

---

## Starter Code Template & Submission Skeleton

```python
"""
pre_rmsnorm_resnet.py - Production Deep Residual MLP Suite
"""
import torch
import torch.nn as nn
import math

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.gamma

class ResMLPBlock(nn.Module):
    def __init__(self, dim: int, dropout_p: float = 0.1):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.fc1 = nn.Linear(dim, dim * 2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim * 2, dim)
        self.drop = nn.Dropout(dropout_p)
        
        # Kaiming init
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.norm(x)
        h = self.fc2(self.act(self.fc1(h)))
        return residual + self.drop(h)

class DeepTabularResNet(nn.Module):
    def __init__(self, in_features: int, hidden_dim: int, num_classes: int, depth: int = 8):
        super().__init__()
        self.proj_in = nn.Linear(in_features, hidden_dim)
        self.blocks = nn.ModuleList([ResMLPBlock(hidden_dim) for _ in range(depth)])
        self.final_norm = RMSNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj_in(x)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        return self.head(x)

# Verification Run
if __name__ == "__main__":
    torch.manual_seed(42)
    B, D_in, D_hidden, Num_Classes = 64, 32, 64, 2
    
    model = DeepTabularResNet(in_features=D_in, hidden_dim=D_hidden, num_classes=Num_Classes, depth=6)
    inputs = torch.randn(B, D_in)
    labels = torch.randint(0, Num_Classes, (B,))
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    
    # Forward & Backward Pass
    model.train()
    logits = model(inputs)
    loss = criterion(logits, labels)
    loss.backward()
    
    # Clip Gradients
    norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    print(f"Initial Loss: {loss.item():.4f}")
    print(f"Gradient Norm (Clipped): {norm:.4f}")
    assert not torch.isnan(loss)
```

---

## Grading Rubric (100 Points Total)

| Component | Points | Criteria |
| :--- | :--- | :--- |
| **Pre-RMSNorm Block Architecture** | 35 | Flawless residual skip connection, RMSNorm implementation, correct Kaiming scaling. |
| **Decoupled AdamW Mechanics** | 25 | Parameterized weight decay independent of gradient moments; correct bias correction. |
| **Gradient Clipping & Stability** | 20 | Bounding gradient norm vectorially; no exploding/vanishing gradients at depth $\ge 8$. |
| **Mixed Precision & Modern PyTorch** | 10 | Proper context manager handling (`autocast`) and clean tensor typing. |
| **Verification & Convergence** | 10 | Loss monotonically decreases over 10 training epochs without numerical NaN values. |
