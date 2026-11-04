# Month 0, Week 4 Cheatsheet: Multivariable Calculus, Convexity & Stochastic Optimization

---

## 1. Gradients, Directional Derivatives & Hessians

### Gradient Vector $\nabla f(x) \in \mathbb{R}^d$
$$\nabla f(x) = \left[ \frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \dots, \frac{\partial f}{\partial x_d} \right]^T$$
- **Direction of Steepest Ascent**: $u^* = \frac{\nabla f(x)}{\|\nabla f(x)\|_2}$.
- **Directional Derivative along unit vector $u$**: $D_u f(x) = \nabla f(x)^T u$.

### Hessian Matrix $H(x) = \nabla^2 f(x) \in \mathbb{R}^{d \times d}$
$$H_{i, j}(x) = \frac{\partial^2 f(x)}{\partial x_i \partial x_j}$$
- **Symmetric**: By Schwarz's theorem (Clairaut), $H_{i, j} = H_{j, i}$ for $C^2$ functions.
- **Local Curvature Classification**:
  - $H(x^*) \succ 0$ (all $\lambda_i > 0$): Strict Local Minimum.
  - $H(x^*) \prec 0$ (all $\lambda_i < 0$): Strict Local Maximum.
  - Indefinite ($\exists \lambda_i > 0, \lambda_j < 0$): Saddle Point.

---

## 2. Second-Order Taylor Polynomial & Condition Numbers

### Taylor Expansion around $x_0$
$$f(x) \approx f(x_0) + \nabla f(x_0)^T (x - x_0) + \frac{1}{2} (x - x_0)^T H(x_0) (x - x_0)$$

### Condition Number $\kappa(H)$ & Pathological Curvature
$$\kappa(H) = \frac{\lambda_{\max}(H)}{\lambda_{\min}(H)}$$
- $\kappa \approx 1$: Isotropic spherical bowl (fast GD convergence).
- $\kappa \gg 1$: Ill-conditioned anisotropic valley (severe GD zigzagging and oscillation).
- Maximum stable learning rate for gradient descent: $\eta < \frac{2}{\lambda_{\max}(H)}$.

---

## 3. Second-Order Optimization: Newton-Raphson

The pure Newton step minimizes the second-order Taylor expansion in a single step:
$$\Delta x_{\text{Newton}} = -H(x)^{-1} \nabla f(x)$$
- **Quadratic Local Convergence**: Error satisfies $\|x_{t+1} - x^*\| \le M \|x_t - x^*\|^2$.
- **Computational Bottleneck**: Computing and inverting an $O(d \times d)$ Hessian costs $\mathcal{O}(d^3)$ per step, making it infeasible for deep networks with millions of parameters.

---

## 4. Modern Stochastic Optimization Algorithms

```
                       Stochastic Optimizer Evolution
    SGD (Pure 1st Order) ──► Polyak Momentum (Damps Oscillation)
                                         │
    AdaGrad (Accumulates $g^2$)          ▼
              │                Nesterov Accelerated (Lookahead Gradient)
              ▼
    RMSProp (Exponential Decay $v_t$)
              │
              ▼
    Adam (Momentum + RMSProp + Bias Correction)
              │
              ▼
    AdamW (Decoupled Weight Decay: Loshchilov & Hutter 2017)
```

### Mathematical Optimizer Updates Table

| Optimizer | Accumulators | Parameter Update Step $\theta_{t+1}$ |
|:---|:---|:---|
| **SGD + Momentum** | $v_t = \beta v_{t-1} + g_t$ | $\theta_t - \eta v_t$ |
| **Nesterov (NAG)** | $v_t = \beta v_{t-1} + \nabla f(\theta_t - \beta v_{t-1})$ | $\theta_t - \eta v_t$ |
| **RMSProp** | $v_t = \alpha v_{t-1} + (1 - \alpha) g_t^2$ | $\theta_t - \frac{\eta}{\sqrt{v_t} + \epsilon} g_t$ |
| **Adam** | $m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$<br>$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$<br>$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$ | $\theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$ |
| **AdamW** | Same bias-corrected $\hat{m}_t, \hat{v}_t$ as Adam | $\theta_t - \eta \lambda \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$ |

---

## 5. Python Reference: Vectorized AdamW Optimizer

```python
import numpy as np

class VectorizedAdamW:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01):
        self.lr = lr
        self.b1, self.b2 = beta1, beta2
        self.eps = eps
        self.wd = weight_decay
        self.m = None
        self.v = None
        self.t = 0

    def step(self, theta: np.ndarray, grad: np.ndarray) -> np.ndarray:
        if self.m is None:
            self.m = np.zeros_like(theta)
            self.v = np.zeros_like(theta)
        self.t += 1
        
        self.m = self.b1 * self.m + (1.0 - self.b1) * grad
        self.v = self.b2 * self.v + (1.0 - self.b2) * (grad ** 2)
        
        m_hat = self.m / (1.0 - self.b1 ** self.t)
        v_hat = self.v / (1.0 - self.b2 ** self.t)
        
        # Decoupled decay + adaptive step
        theta = theta - self.lr * self.wd * theta
        theta = theta - (self.lr / (np.sqrt(v_hat) + self.eps)) * m_hat
        return theta
```
