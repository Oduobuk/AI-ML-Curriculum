Week 4: Essential Math Part 2 Cheatsheet

Quick reference for vectors, matrices, linear algebra, and calculus.

🎯 Vectors

Creating & Operations
```python
import numpy as np

Create vectors
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])

Element-wise operations
v1 + v2          # [5, 7, 9]
v1 - v2          # [-3, -3, -3]
v1 * 2           # [2, 4, 6]
```

Vector Magnitude
```python
magnitude = np.linalg.norm(v1)  # √(1² + 2² + 3²)
```

Dot Product
```python
dot = np.dot(v1, v2)     # 1×4 + 2×5 + 3×6 = 32
dot = v1 @ v2            # Same thing
```

Perpendicular Vectors
```python
If v1 · v2 = 0, they're perpendicular
if np.dot(v1, v2) == 0:
    print("Perpendicular")
```

📊 Matrices

Creating & Accessing
```python
M = np.array([[1, 2, 3],
              [4, 5, 6]])

M.shape       # (2, 3)
M[0, 0]       # 1 (row 0, col 0)
M[0, :]       # [1, 2, 3] (first row)
M[:, 0]       # [1, 4] (first column)
```

Basic Operations
```python
Addition
A + B

Scalar multiplication
A * 2

Matrix multiplication (important!)
A @ B
np.dot(A, B)
```

Matrix Properties
```python
Transpose
A.T

Determinant
det = np.linalg.det(A)

Inverse (if square and non-singular)
A_inv = np.linalg.inv(A)

Trace (sum of diagonal)
trace = np.trace(A)
```

Special Matrices
```python
Identity matrix
I = np.eye(3)

Zeros
Z = np.zeros((3, 3))

Ones
O = np.ones((3, 3))
```

🔄 Linear Transformations

Rotation (2D)
```python
angle = np.pi / 4  # 45 degrees
R = np.array([[np.cos(angle), -np.sin(angle)],
              [np.sin(angle), np.cos(angle)]])

v_rotated = R @ v
```

Scaling
```python
S = np.array([[2, 0],      # Scale x by 2
              [0, 0.5]])   # Scale y by 0.5

v_scaled = S @ v
```

Projection
```python
Project v onto u
u = np.array([1, 0])
v = np.array([3, 4])
projection = (np.dot(v, u) / np.dot(u, u)) * u
```

📐 Calculus

Numerical Derivative
```python
def derivative(f, x, h=0.0001):
    return (f(x + h) - f(x - h)) / (2 * h)

def f(x):
    return x2

derivat2 = derivative(f, 2)  # ≈ 4
```

Symbolic Derivatives (Common Rules)
```
f(x) = x^n      →  f'(x) = n·x^(n-1)
f(x) = e^x      →  f'(x) = e^x
f(x) = ln(x)    →  f'(x) = 1/x
f(x) = sin(x)   →  f'(x) = cos(x)
f(x) = cos(x)   →  f'(x) = -sin(x)
```

Partial Derivatives
```python
def f(x, y):
    return x2 + y2

∂f/∂x
def df_dx(x, y, h=0.0001):
    return (f(x + h, y) - f(x - h, y)) / (2 * h)

∂f/∂y
def df_dy(x, y, h=0.0001):
    return (f(x, y + h) - f(x, y - h)) / (2 * h)
```

Gradient (Vector of Partial Derivatives)
```python
def gradient(f, x, y, h=0.0001):
    gx = df_dx(f, x, y, h)
    gy = df_dy(f, x, y, h)
    return np.array([gx, gy])
```

🚀 Optimization

Gradient Descent
```python
def gradientdescent(f, df, x0, learningrate=0.01, iterations=100):
    x = x0
    for i in range(iterations):
        gradient = df(x)
        x = x - learning_rate * gradient
    return x

Use it
minx = gradientdescent(f, df, x0=0)
```

Learning Rate Impact
```
- Too small: Slow convergence
- Too large: May diverge
- Optimal: Sweet spot (usually 0.01 - 0.1)
```

Convergence Check
```python
if abs(x_new - x) < threshold:
    print("Converged!")
    break
```

🔗 Connecting to ML

| Concept | ML Application |
|---------|-----------------|
| Vectors | Input features, embeddings |
| Matrices | Weight matrices, image data |
| Dot Product | Similarity, neuron activation |
| Matrix Mult | Neural network layers |
| Derivatives | Computing gradients |
| Gradient Descent | Training algorithm |

📦 Common Libraries

```python
import numpy as np          # Numerical computing
from numpy.linalg import * # Linear algebra

import scipy.optimize       # Optimization
import matplotlib.pyplot as plt  # Visualization
```

[Back to Lecture](../lecturenotes/Week4Math_Part2.md)
