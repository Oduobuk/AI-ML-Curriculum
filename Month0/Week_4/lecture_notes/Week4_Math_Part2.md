Week 4: Essential Math for AI/ML (Part 2)

Learning Objectives

By the end of this week, you will be able to:
- [ ] Understand vectors and their operations
- [ ] Perform matrix operations
- [ ] Compute dot products and understand their geometric meaning
- [ ] Understand linear transformations
- [ ] Grasp calculus concepts: derivatives and rates of change
- [ ] Understand optimization and why it matters
- [ ] Connect math concepts to machine learning

1. Vectors: Direction and Magnitude

1.1 What is a Vector?

A vector is a quantity with both magnitude (how big) and direction.

Visual Examples:
- Arrow pointing northeast with length 5: →↗
- Displacement: "Move 3 units right, 2 units up"
- Velocity: "Move 60 mph east"

Mathematical Notation:
```
v = (3, 2)          # 2D vector: 3 right, 2 up
v = (1, 2, 3)       # 3D vector
v = (x₁, x₂, ..., xₙ)  # n-dimensional vector
```

1.2 Vector Operations in Python

```python
import numpy as np

Create vectors
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])

Vector addition (add components)
result = v1 + v2    # [5, 7, 9]

Vector subtraction
result = v1 - v2    # [-3, -3, -3]

Scalar multiplication
result = v1 * 2     # [2, 4, 6]

Vector magnitude (length)
magnitude = np.linalg.norm(v1)  # √(1² + 2² + 3²) ≈ 3.74
```

1.3 Dot Product (Inner Product)

The dot product combines two vectors into a single number.

```
a · b = (a₁ × b₁) + (a₂ × b₂) + ... + (aₙ × bₙ)
```

Example:
```
[1, 2] · [3, 4] = (1×3) + (2×4) = 3 + 8 = 11
```

In Python:
```python
import numpy as np

a = np.array([1, 2])
b = np.array([3, 4])

dot_product = np.dot(a, b)  # 11
or
dot_product = a @ b         # 11 (shorthand)
```

Geometric Interpretation:
```
a · b = |a| × |b| × cos(θ)

where θ is the angle between vectors
```

- If dot product = 0: vectors are perpendicular (90° angle)
- If dot product > 0: vectors point in similar direction
- If dot product < 0: vectors point in opposite direction

Why it matters for ML:
- Neural networks use dot products in every layer
- Similarity measurements use dot products
- Support Vector Machines use dot products

1.4 Vector Visualization

```python
import matplotlib.pyplot as plt
import numpy as np

Create vectors
v1 = np.array([3, 2])
v2 = np.array([1, 4])

Plot vectors as arrows from origin
plt.figure(figsize=(8, 8))
plt.quiver(0, 0, v1[0], v1[1], angles='xy', scale_units='xy', scale=1, color='red', label='v1')
plt.quiver(0, 0, v2[0], v2[1], angles='xy', scale_units='xy', scale=1, color='blue', label='v2')

Set axis limits
plt.xlim(-1, 5)
plt.ylim(-1, 5)
plt.axhline(y=0, color='k', linewidth=0.5)
plt.axvline(x=0, color='k', linewidth=0.5)
plt.grid(True)
plt.legend()
plt.title("Vectors v1 and v2")
plt.show()

Dot product
dot = np.dot(v1, v2)  # 3×1 + 2×4 = 11
print(f"v1 · v2 = {dot}")
```

2. Matrices: Arrays of Numbers

2.1 What is a Matrix?

A matrix is a rectangular grid of numbers with rows and columns.

```
    [1  2  3]
A = [4  5  6]

This is a 2×3 matrix (2 rows, 3 columns)
```

In Python:
```python
import numpy as np

A = np.array([[1, 2, 3],
              [4, 5, 6]])

print(A.shape)    # (2, 3)
print(A[0])       # [1 2 3] (first row)
print(A[:, 0])    # [1 4] (first column)
print(A[0, 1])    # 2 (element at row 0, col 1)
```

2.2 Matrix Operations

Addition (element-wise):
```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

C = A + B  # [[6, 8], [10, 12]]
```

Scalar Multiplication:
```python
A = np.array([[1, 2], [3, 4]])
C = A * 2  # [[2, 4], [6, 8]]
```

Matrix Multiplication:
```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

C = A @ B  # Matrix multiplication
[1×5 + 2×7,  1×6 + 2×8]   [19, 22]
[3×5 + 4×7,  3×6 + 4×8] = [43, 50]
```

Important: Matrix multiplication is NOT element-wise!

Transpose (flip rows and columns):
```python
A = np.array([[1, 2, 3],
              [4, 5, 6]])

A_T = A.T  # [[1, 4],
           #  [2, 5],
           #  [3, 6]]
```

2.3 Special Matrices

Identity Matrix (like multiplying by 1):
```python
I = np.eye(3)  # [[1, 0, 0],
               #  [0, 1, 0],
               #  [0, 0, 1]]

A @ I == A  # Always true
```

Zero Matrix:
```python
Z = np.zeros((2, 3))  # All zeros
```

2.4 Matrix Determinant and Inverse

Determinant: Scalar value that tells us about the matrix

```python
A = np.array([[1, 2], [3, 4]])
det = np.linalg.det(A)  # -2

If det = 0, matrix is singular (has no inverse)
```

Inverse: The "opposite" of a matrix

```python
A = np.array([[1, 2], [3, 4]])
A_inv = np.linalg.inv(A)

A @ A_inv = Identity matrix
```

3. Linear Transformations

A linear transformation takes input vectors and produces output vectors using matrix multiplication.

3.1 Geometric Meaning

```python
import matplotlib.pyplot as plt
import numpy as np

Transformation matrix
T = np.array([[2, 0],      # Scale x by 2
              [0, 1]])     # Keep y same

Original vectors
v1 = np.array([1, 0])
v2 = np.array([0, 1])
v3 = np.array([1, 1])

Apply transformation
v1_transformed = T @ v1  # [2, 0]
v2_transformed = T @ v2  # [0, 1]
v3_transformed = T @ v3  # [2, 1]

Plot original and transformed
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.quiver(0, 0, v1[0], v1[1], color='red', scale=3)
plt.quiver(0, 0, v2[0], v2[1], color='blue', scale=3)
plt.quiver(0, 0, v3[0], v3[1], color='green', scale=3)
plt.title("Original Vectors")
plt.xlim(-1, 3)
plt.ylim(-1, 3)
plt.grid(True)
plt.axhline(y=0)
plt.axvline(x=0)

plt.subplot(1, 2, 2)
plt.quiver(0, 0, v1transformed[0], v1transformed[1], color='red', scale=3)
plt.quiver(0, 0, v2transformed[0], v2transformed[1], color='blue', scale=3)
plt.quiver(0, 0, v3transformed[0], v3transformed[1], color='green', scale=3)
plt.title("After Transformation (Scaled x by 2)")
plt.xlim(-1, 3)
plt.ylim(-1, 3)
plt.grid(True)
plt.axhline(y=0)
plt.axvline(x=0)

plt.tight_layout()
plt.show()
```

3.2 Common Transformations

Rotation:
```python
Rotate by 45 degrees
angle = np.pi / 4  # 45 degrees in radians
R = np.array([[np.cos(angle), -np.sin(angle)],
              [np.sin(angle), np.cos(angle)]])
```

Scaling:
```python
Scale x by 2, y by 0.5
S = np.array([[2, 0],
              [0, 0.5]])
```

Projection:
```python
Project onto x-axis
P = np.array([[1, 0],
              [0, 0]])
```

4. Calculus Basics: Derivatives

4.1 What is a Derivative?

A derivative measures the rate of change of a function—how fast it's going up or down.

Visual Example:
```
If y = x²
Derivative tells us the slope at any point

At x = 1: slope = 2
At x = 2: slope = 4
At x = 3: slope = 6
```

4.2 Computing Derivatives

Simple rules:

```
f(x) = x²      → f'(x) = 2x
f(x) = x³      → f'(x) = 3x²
f(x) = sin(x)  → f'(x) = cos(x)
f(x) = eˣ      → f'(x) = eˣ
```

In Python:
```python
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

Define a function
def f(x):
    return x2

Approximate derivative numerically
def derivative(f, x, h=0.0001):
    return (f(x + h) - f(x - h)) / (2 * h)

Test
x = 2
deriv = derivative(f, x)
print(f"Derivative at x={x}: {deriv:.4f}")  # ≈ 4

Plot
x_vals = np.linspace(-3, 3, 100)
yvals = f(xvals)

plt.plot(xvals, yvals, label='f(x) = x²')
plt.plot(x, f(x), 'ro', label=f'Point ({x}, {f(x)})')
plt.title('Function and Derivative')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True)
plt.legend()
plt.show()
```

4.3 Partial Derivatives

With multiple variables, we compute the derivative with respect to one variable.

```
f(x, y) = x² + y²

∂f/∂x = 2x  (derivative with respect to x, treating y as constant)
∂f/∂y = 2y  (derivative with respect to y, treating x as constant)
```

In Python:
```python
def f(x, y):
    return x2 + y2

Partial derivative with respect to x
def df_dx(x, y, h=0.0001):
    return (f(x + h, y) - f(x - h, y)) / (2 * h)

Partial derivative with respect to y
def df_dy(x, y, h=0.0001):
    return (f(x, y + h) - f(x, y - h)) / (2 * h)

Test
print(f"∂f/∂x at (2, 3): {df_dx(2, 3):.4f}")  # ≈ 4
print(f"∂f/∂y at (2, 3): {df_dy(2, 3):.4f}")  # ≈ 6
```

5. Optimization: Finding Minimums and Maximums

5.1 Why Optimization Matters for ML

Machine learning is optimization!

We want to find parameters (weights) that minimize error:
```
Minimize: Loss = (Predicted - Actual)²
```

5.2 Gradient Descent

Gradient descent is an algorithm that finds the minimum of a function.

```
1. Start at a random point
2. Compute the gradient (derivative)
3. Move in the opposite direction of the gradient
4. Repeat until converged
```

Intuition: Like walking downhill in fog—always take a step down.

```python
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return (x - 3)2 + 2  # Minimum at x=3

def df(x):
    return 2 * (x - 3)

Gradient descent
x = 0
learning_rate = 0.1
history = [x]

for i in range(20):
    x = x - learning_rate * df(x)
    history.append(x)

print(f"Final x: {x:.4f}")  # Should be close to 3

Plot
x_vals = np.linspace(-2, 8, 100)
yvals = f(xvals)

plt.plot(xvals, yvals, 'b-', label='f(x)')
plt.plot(history, f(np.array(history)), 'r.-', label='Gradient Descent Path')
plt.plot(3, f(3), 'g*', markersize=15, label='Minimum')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Gradient Descent')
plt.legend()
plt.grid(True)
plt.show()
```

5.3 Optimization with Multiple Variables

```python
from scipy.optimize import minimize
import numpy as np

Function to minimize
def f(x):
    return (x[0] - 3)2 + (x[1] - 2)2

Initial guess
x0 = np.array([0, 0])

Minimize
result = minimize(f, x0, method='BFGS')

print(f"Optimal x: {result.x}")        # [3, 2]
print(f"Minimum value: {result.fun}")  # ≈ 0
```

6. Connecting to Machine Learning

6.1 Linear Regression

```python
import numpy as np
from sklearn.linear_model import LinearRegression

Data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

Fit model (finds optimal weights)
model = LinearRegression()
model.fit(X, y)

print(f"Weight: {model.coef_}")      # 2
print(f"Intercept: {model.intercept_}")  # 0

Predictions
predictions = model.predict([[6]])
print(f"Prediction at x=6: {predictions}")  # [12]
```

6.2 Neural Networks

Neural networks are matrices multiplied by activation functions:

```
output = relu(input @ weights + bias)
```

The training process uses derivatives to update weights via gradient descent!

7. Summary: Math in ML

| Concept | ML Application |
|---------|-----------------|
| Vectors | Input features, weights |
| Matrices | Images, weight matrices in networks |
| Dot Products | Feature similarity, neural network layers |
| Derivatives | Gradient descent, backpropagation |
| Optimization | Training models (finding best weights) |

🎯 Key Takeaways

- Vectors have direction and magnitude
- Matrices are grids of numbers that transform vectors
- Dot products measure similarity
- Derivatives measure rate of change
- Gradient descent finds minimums (used in ML training)
- Optimization is the core of machine learning

📚 Additional Resources

- [3Blue1Brown: Essence of Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)
- [3Blue1Brown: Essence of Calculus](https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr28EL8nRrJ7wMCo2VGES)
- [Khan Academy: Linear Algebra](https://www.khanacademy.org/math/linear-algebra)
- [Khan Academy: Calculus](https://www.khanacademy.org/math/calculus-1)

Ready for exercises? → [Week 4 Exercises](../exercises/Week4_Exercises.md)
