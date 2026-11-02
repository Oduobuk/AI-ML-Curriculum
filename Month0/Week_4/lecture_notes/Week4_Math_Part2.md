# Month 0, Week 4: Calculus, Autodiff & Probability for Machine Learning

![Gradient Descent Optimization](../../../docs/images/gradient_descent_intuition.png)

## 1. The Calculus of Optimization: Derivatives & Gradients

Machine learning is fundamentally an optimization problem: we define a **Loss Function** $L(\mathbf{w})$ measuring how wrong our model is, and use calculus to systematically adjust parameters $\mathbf{w}$ to minimize that loss.

### What is a Derivative?
The instantaneous rate of change of a function $f(x)$ with respect to $x$:
$$f'(x) = \frac{df}{dx} = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$$

* **Positive Slope ($f'(x) > 0$):** As $x$ increases, $f(x)$ increases. To decrease $f(x)$, move $x$ **left** (subtract).
* **Negative Slope ($f'(x) < 0$):** As $x$ increases, $f(x)$ decreases. To decrease $f(x)$, move $x$ **right** (add).

### Common Differentiation Rules:
1. **Power Rule:** $\frac{d}{dx}[x^n] = n x^{n-1}$
2. **Constant Multiple:** $\frac{d}{dx}[c \cdot f(x)] = c \cdot f'(x)$
3. **Sum Rule:** $\frac{d}{dx}[f(x) + g(x)] = f'(x) + g'(x)$
4. **Exponential:** $\frac{d}{dx}[e^x] = e^x$

### The Chain Rule (Foundation of Neural Network Backpropagation):
If $y = f(u)$ and $u = g(x)$, then:
$$\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}$$

---

## 2. Gradient Descent Algorithm from Scratch

A gradient is a vector containing all the partial derivatives of a multivariable function:
$$\nabla f(\mathbf{w}) = \left[ \frac{\partial f}{\partial w_1}, \frac{\partial f}{\partial w_2}, \dots, \frac{\partial f}{\partial w_d} \right]^T$$

The gradient points in the direction of **steepest ascent**. Therefore, we step in the **opposite** direction to minimize error:
$$w \leftarrow w - \alpha \frac{\partial L}{\partial w}$$

Where $\alpha$ (alpha) is the **learning rate** (hyperparameter).

```python
# Minimizing J(w) = w^2 using Gradient Descent from scratch
# Analytical derivative: dJ/dw = 2*w

def gradient_descent_1d(initial_w: float, learning_rate: float, epochs: int):
    w = initial_w
    print(f"Starting at w = {w:.4f}, Loss = {w**2:.4f}")
    
    for epoch in range(1, epochs + 1):
        grad = 2 * w            # Compute derivative
        w = w - learning_rate * grad  # Parameter update
        loss = w ** 2
        
        if epoch % 5 == 0 or epoch == epochs:
            print(f"Epoch {epoch:02d}: w = {w:.6f}, Loss = {loss:.8f}")
            
    return w

optimal_w = gradient_descent_1d(initial_w=5.0, learning_rate=0.2, epochs=25)
print(f"Optimization complete! Converged near minimum: {optimal_w:.6f}")
```

---

## 3. Probability Foundations for AI

Classification models output probabilistic confidences (e.g., Softmax / Sigmoid).

### Expected Value (Mean) & Variance:
$$\mu = \mathbb{E}[X] = \frac{1}{N} \sum_{i=1}^N x_i$$
$$\sigma^2 = \text{Var}(X) = \frac{1}{N} \sum_{i=1}^N (x_i - \mu)^2$$

### Bayes' Theorem:
The mathematical rule for updating hypotheses based on observed data:
$$P(A \mid B) = \frac{P(B \mid A) \cdot P(A)}{P(B)}$$

* $P(A \mid B)$: Posterior probability (probability of class $A$ given feature evidence $B$).
* $P(B \mid A)$: Likelihood (probability of seeing evidence $B$ given class $A$).
* $P(A)$: Prior probability of class $A$.
* $P(B)$: Total evidence probability.

---

## Hands-On Assignment
Navigate to [Week 4 Assignment](../assignments/Week4_Assignment.md) to implement a complete 2D Gradient Descent optimizer with interactive matplotlib trajectory tracing.
