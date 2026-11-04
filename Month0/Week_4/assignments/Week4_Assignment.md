# Week 4 Production Engineering Assignment: High-Dimensional Stochastic Optimization Suite & Non-Convex Landscape Benchmark

## Executive Overview & Architectural Motivation
Deep neural networks optimize non-convex, high-dimensional empirical risk surfaces characterised by pathological curvature, ill-conditioned ravines, saddle points, and stochastic batch gradient noise.

In this assignment, you will architect an industrial-grade **Stochastic Optimization Benchmark Suite** in pure Python and NumPy. You will build, calibrate, and mathematically evaluate four first-order optimizers:
1. **SGD with Polyak Heavy-Ball Momentum**
2. **Nesterov Accelerated Gradient (NAG)**
3. **RMSProp with Exponential Variance Decay**
4. **AdamW (Decoupled Weight Decay Adam)**

You will evaluate these optimizers against classical pathological test functions (Rosenbrock Valley, Beale Function, Rastrigin Multimodal Surface) and analyze their convergence trajectories, oscillation dampening, and saddle point escape properties.

---

## Technical Specifications & Mathematical Formulation

### 1. Rosenbrock Valley (Ill-conditioned non-convex banana)
$$f(x, y) = (a - x)^2 + b(y - x^2)^2 \quad (a=1, b=100)$$
- Global minimum at $(1, 1)$ where $f(1, 1) = 0$.

### 2. Beale Function (Sharp contours and flat corners)
$$f(x, y) = (1.5 - x + xy)^2 + (2.25 - x + xy^2)^2 + (2.625 - x + xy^3)^2$$
- Global minimum at $(3, 0.5)$ where $f(3, 0.5) = 0$.

### 3. Decoupled Weight Decay (AdamW) Update Rule
Unlike L2-regularized Adam where weight decay is added directly to gradient $g_t$ (distorting second-moment statistics), AdamW applies weight decay directly to parameter updates:
$$\theta_t = \theta_{t-1} - \eta \lambda \theta_{t-1} - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

---

## Detailed Deliverable Requirements

### 1. Generic Optimizer Protocol (`optimizers.py`)
Implement an abstract base class `BaseOptimizer`:
```python
class BaseOptimizer(ABC):
    @abstractmethod
    def step(self, param: np.ndarray, grad: np.ndarray) -> np.ndarray:
        pass
    @abstractmethod
    def reset(self) -> None:
        pass
```
Implement subclasses:
- `SGDMomentum(lr, beta)`
- `NesterovSGD(lr, beta)`
- `RMSProp(lr, alpha, eps)`
- `AdamW(lr, beta1, beta2, weight_decay, eps)`

### 2. Analytical Loss Landscapes & Gradient Engine (`landscapes.py`)
Implement vectorized evaluations and exact analytical gradients for:
- Rosenbrock function and gradient.
- Beale function and gradient.
- Rastrigin function and gradient.

### 3. Benchmark Runner & Phase-Space Analysis (`benchmark.py`)
- Execute all optimizers across 500 iterations from identical starting coordinates ($x_0 = [-1.5, 2.0]$ for Rosenbrock).
- Measure:
  1. Final distance to global optimum $\|x^* - \hat{x}\|_2$.
  2. Loss reduction velocity (epoch to reach loss $< 10^{-2}$).
  3. Total path length (measuring trajectory oscillation):
     $$\mathcal{P} = \sum_{t=1}^T \|x_t - x_{t-1}\|_2$$

---

## Starter Code Template & Submission Skeleton

```python
"""
optimization_benchmark.py - Production Stochastic Optimizer Suite
"""
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Tuple

class BaseOptimizer(ABC):
    @abstractmethod
    def step(self, param: np.ndarray, grad: np.ndarray) -> np.ndarray:
        pass
    @abstractmethod
    def reset(self) -> None:
        pass

class AdamW(BaseOptimizer):
    def __init__(self, lr: float = 0.01, beta1: float = 0.9, beta2: float = 0.999, 
                 weight_decay: float = 0.01, eps: float = 1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.weight_decay = weight_decay
        self.eps = eps
        self.m: Optional[np.ndarray] = None
        self.v: Optional[np.ndarray] = None
        self.t: int = 0

    def reset(self) -> None:
        self.m = None
        self.v = None
        self.t = 0

    def step(self, param: np.ndarray, grad: np.ndarray) -> np.ndarray:
        if self.m is None:
            self.m = np.zeros_like(param)
            self.v = np.zeros_like(param)
        self.t += 1
        
        # Exponential moving averages
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * (grad ** 2)
        
        # Bias correction
        m_hat = self.m / (1.0 - self.beta1 ** self.t)
        v_hat = self.v / (1.0 - self.beta2 ** self.t)
        
        # Decoupled weight decay step
        decayed_param = param - self.lr * self.weight_decay * param
        
        # Adaptive gradient step
        update = (self.lr / (np.sqrt(v_hat) + self.eps)) * m_hat
        return decayed_param - update

class SGDMomentum(BaseOptimizer):
    def __init__(self, lr: float = 0.001, momentum: float = 0.9):
        self.lr = lr
        self.momentum = momentum
        self.velocity: Optional[np.ndarray] = None

    def reset(self) -> None:
        self.velocity = None

    def step(self, param: np.ndarray, grad: np.ndarray) -> np.ndarray:
        if self.velocity is None:
            self.velocity = np.zeros_like(param)
        self.velocity = self.momentum * self.velocity + grad
        return param - self.lr * self.velocity

# Verification Test Run
if __name__ == "__main__":
    # Test on 2D Rosenbrock function
    def rosenbrock_fg(x: np.ndarray) -> Tuple[float, np.ndarray]:
        loss = (1.0 - x[0])**2 + 100.0 * (x[1] - x[0]**2)**2
        grad = np.array([
            -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0]**2),
            200.0 * (x[1] - x[0]**2)
        ])
        return loss, grad

    optimizers: Dict[str, BaseOptimizer] = {
        "SGD_Momentum": SGDMomentum(lr=0.001, momentum=0.9),
        "AdamW": AdamW(lr=0.05, weight_decay=0.0)
    }

    start_pos = np.array([-1.2, 1.0])
    print("Testing Optimizer Convergence on Rosenbrock Valley:")
    for name, opt in optimizers.items():
        pos = start_pos.copy()
        opt.reset()
        for i in range(1000):
            l, g = rosenbrock_fg(pos)
            pos = opt.step(pos, g)
        final_loss, _ = rosenbrock_fg(pos)
        print(f"[{name}] Final Position: {np.round(pos, 4)} | Final Loss: {final_loss:.6f}")
        assert np.linalg.norm(pos - np.array([1.0, 1.0])) < 0.5
```

---

## Grading Rubric (100 Points Total)

| Component | Points | Criteria |
| :--- | :--- | :--- |
| **Mathematical Correctness of Optimizers** | 35 | Flawless implementation of SGD+Momentum, NAG lookahead, RMSProp, and AdamW bias-correction. |
| **Analytical Gradient Derivations** | 25 | Exact derivation and vectorized computation of Rosenbrock, Beale, and Rastrigin gradients. |
| **Trajectory & Oscillation Analysis** | 20 | Quantitative calculation of path length $\mathcal{P}$, loss curves, and condition-number sensitivity. |
| **Code Architecture & Typing** | 10 | Adherence to abstract base classes, complete type hints, docstrings, and modularity. |
| **Verification & Reproducibility** | 10 | Seeded execution reproducing identical optimization trajectories across runs. |
