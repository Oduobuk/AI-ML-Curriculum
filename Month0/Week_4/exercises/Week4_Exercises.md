# Week 4: Graduate-Level Problem Set — Multivariable Calculus, Optimization & Convex Geometry

Welcome to the Week 4 Problem Set. This suite rigorous develops the multivariable calculus and convex optimization theoretical foundation necessary for understanding deep learning loss landscapes, condition numbers, second-order optimization, and stochastic gradient dynamics.

---

## Tier 1: Multivariable Calculus, Gradients & Hessians

### Exercise 1: Gradient Vector & Hessian Matrix Derivation
**Concepts**: Gradient $\nabla f(x)$, Hessian matrix $\nabla^2 f(x)$, positive semi-definiteness, Taylor polynomial approximations.

#### Problem Specification
Consider the non-convex Rosenbrock valley function in $\mathbb{R}^2$:
$$f(x_1, x_2) = (1 - x_1)^2 + 100(x_2 - x_1^2)^2$$
1. Analytically derive the gradient vector $\nabla f(x) \in \mathbb{R}^2$.
2. Analytically derive the $2 \times 2$ Hessian matrix $H(x) = \nabla^2 f(x)$.
3. Evaluate the Hessian at the global minimum $x^* = (1, 1)^T$. Compute its eigenvalues $\lambda_1, \lambda_2$ and the condition number $\kappa(H) = \frac{\lambda_{\max}}{\lambda_{\min}}$.
4. Construct the second-order Taylor approximation $f_{\text{quad}}(x)$ around $x^*$ and measure approximation error at $x = (1.05, 1.05)^T$.

#### Verification Solution
```python
import numpy as np

def rosenbrock(x: np.ndarray) -> float:
    return float((1.0 - x[0])**2 + 100.0 * (x[1] - x[0]**2)**2)

def rosenbrock_grad(x: np.ndarray) -> np.ndarray:
    df_dx1 = -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0]**2)
    df_dx2 = 200.0 * (x[1] - x[0]**2)
    return np.array([df_dx1, df_dx2])

def rosenbrock_hessian(x: np.ndarray) -> np.ndarray:
    d2f_dx1_2 = 2.0 - 400.0 * x[1] + 1200.0 * (x[0]**2)
    d2f_dx1_dx2 = -400.0 * x[0]
    d2f_dx2_2 = 200.0
    return np.array([
        [d2f_dx1_2, d2f_dx1_dx2],
        [d2f_dx1_dx2, d2f_dx2_2]
    ])

# Verification Run
if __name__ == "__main__":
    x_star = np.array([1.0, 1.0])
    grad_star = rosenbrock_grad(x_star)
    H_star = rosenbrock_hessian(x_star)
    
    assert np.allclose(grad_star, [0.0, 0.0]), "Gradient at x* must be zero"
    eigenvalues = np.linalg.eigvalsh(H_star)
    cond_number = eigenvalues[-1] / eigenvalues[0]
    
    print("Hessian at x*:\n", H_star)
    print(f"Eigenvalues: {eigenvalues}")
    print(f"Condition Number kappa(H): {cond_number:.2f}")
    assert eigenvalues[0] > 0, "Hessian must be strictly positive definite at local minimum"
```

---

### Exercise 2: Directional Derivatives & Steepest Descent Proof
**Concepts**: Directional derivative $D_u f(x) = \nabla f(x)^T u$, Cauchy-Schwarz inequality, steepest descent direction.

#### Problem Specification
1. Let $f: \mathbb{R}^d \to \mathbb{R}$ be continuously differentiable. Prove using Cauchy-Schwarz that for any unit vector $u \in \mathbb{R}^d$ ($\|u\|_2 = 1$), the directional derivative satisfies:
   $$D_u f(x) \ge -\|\nabla f(x)\|_2$$
   with equality if and only if $u = -\frac{\nabla f(x)}{\|\nabla f(x)\|_2}$.
2. Implement a numerical verification of this property across 1,000 random unit vectors for a multi-dimensional quadratic bowl $f(x) = \frac{1}{2} x^T A x$.

#### Verification Solution
```python
import numpy as np

def verify_steepest_descent():
    np.random.seed(42)
    d = 5
    M = np.random.randn(d, d)
    A = M.T @ M + np.eye(d)  # Strictly positive definite
    x = np.random.randn(d)
    
    grad = A @ x
    grad_norm = np.linalg.norm(grad)
    steepest_unit = -grad / grad_norm
    min_dir_deriv = np.dot(grad, steepest_unit)
    
    # Sample 1000 random unit vectors
    rand_dirs = np.random.randn(1000, d)
    rand_dirs /= np.linalg.norm(rand_dirs, axis=1, keepdims=True)
    all_dir_derivs = rand_dirs @ grad
    
    assert np.isclose(min_dir_deriv, -grad_norm)
    assert np.all(all_dir_derivs >= min_dir_deriv - 1e-12)
    print(f"Theoretical lower bound -||grad||: {-grad_norm:.6f}")
    print(f"Observed minimum directional deriv:  {np.min(all_dir_derivs):.6f}")

if __name__ == "__main__":
    verify_steepest_descent()
```

---

## Tier 2: 2nd-Order Optimization & Newton-Raphson

### Exercise 3: Newton-Raphson vs Gradient Descent on Ill-Conditioned Objectives
**Concepts**: Newton step $\Delta x = -H(x)^{-1} \nabla f(x)$, quadratic convergence, step-size overshoot.

#### Problem Specification
Given the quadratic bowl $f(x) = \frac{1}{2} x^T A x$ where $A = \begin{bmatrix} 100 & 0 \\ 0 & 1 \end{bmatrix}$ (condition number $\kappa = 100$):
1. Implement standard Gradient Descent with step size $\eta = \frac{1}{\lambda_{\max}} = 0.01$.
2. Implement Newton-Raphson optimization.
3. Compare iterations to convergence ($\|x_k\|_2 < 10^{-6}$) starting from $x_0 = [10.0, 10.0]^T$.

#### Verification Solution
```python
import numpy as np

def run_gd_vs_newton():
    A = np.diag([100.0, 1.0])
    x0 = np.array([10.0, 10.0])
    
    # 1. Gradient Descent
    x_gd = x0.copy()
    eta = 1.0 / 100.0  # Optimal step for fastest dimension
    gd_steps = 0
    while np.linalg.norm(x_gd) > 1e-6 and gd_steps < 5000:
        grad = A @ x_gd
        x_gd = x_gd - eta * grad
        gd_steps += 1
        
    # 2. Newton-Raphson
    x_newton = x0.copy()
    newton_steps = 0
    while np.linalg.norm(x_newton) > 1e-6 and newton_steps < 10:
        grad = A @ x_newton
        H = A
        x_newton = x_newton - np.linalg.solve(H, grad)
        newton_steps += 1
        
    print(f"Gradient Descent steps to convergence: {gd_steps}")
    print(f"Newton-Raphson steps to convergence:   {newton_steps}")
    assert newton_steps == 1, "Newton-Raphson must solve quadratic bowl in exactly 1 step"

if __name__ == "__main__":
    run_gd_vs_newton()
```

---

## Tier 3: Stochastic Optimization Dynamics & Adaptive Moment Estimation

### Exercise 4: Momentum & Adam Optimizer Step Dynamics
**Concepts**: Heavy-ball momentum, exponentially decaying first/second moments, bias correction.

#### Problem Specification
1. Implement pure NumPy update routines for:
   - Polyak Heavy-Ball Momentum: $v_{t+1} = \beta v_t + \nabla f(\theta_t)$, $\theta_{t+1} = \theta_t - \eta v_{t+1}$.
   - Adam: $m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$, $v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$, $\hat{m}_t = \frac{m_t}{1 - \beta_1^t}$, $\hat{v}_t = \frac{v_t}{1 - \beta_2^t}$, $\theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$.
2. Demonstrate how Adam normalizes gradients across disparate coordinate scales on the ill-conditioned elliptical bowl.

#### Verification Solution
```python
import numpy as np

class AdamFromScratch:
    def __init__(self, lr: float = 0.1, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None
        self.v = None
        self.t = 0

    def step(self, param: np.ndarray, grad: np.ndarray) -> np.ndarray:
        if self.m is None:
            self.m = np.zeros_like(param)
            self.v = np.zeros_like(param)
            
        self.t += 1
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * (grad ** 2)
        
        # Bias correction
        m_hat = self.m / (1.0 - self.beta1 ** self.t)
        v_hat = self.v / (1.0 - self.beta2 ** self.t)
        
        return param - (self.lr / (np.sqrt(v_hat) + self.eps)) * m_hat

# Verification Run
if __name__ == "__main__":
    opt = AdamFromScratch(lr=0.05)
    A = np.diag([100.0, 1.0])
    theta = np.array([10.0, 10.0])
    
    for step in range(150):
        grad = A @ theta
        theta = opt.step(theta, grad)
        
    print("Final parameter values after 150 Adam steps:", theta)
    assert np.linalg.norm(theta) < 0.1
```
