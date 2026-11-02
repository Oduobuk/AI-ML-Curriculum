# Month 0, Week 3: Linear Algebra & Tensors from Scratch

![Tensor Broadcasting](../../../docs/images/tensor_broadcasting.png)

## 1. Why Linear Algebra Powers AI
Every image, audio waveform, text token, and neural network weight is represented as an array of numbers. Linear algebra provides the mathematical rules for transforming, projecting, and multiplying these numbers in parallel across GPU threads.

---

## 2. Vectors, Magnitudes & Cosine Similarity

A vector is an ordered list of numbers representing a point or direction in space.

### Vector Magnitude (Euclidean Norm / $L_2$ Norm):
The length of a vector $\mathbf{v} = [v_1, v_2, \dots, v_n]$:
$$\|\mathbf{v}\|_2 = \sqrt{\sum_{i=1}^n v_i^2}$$

### The Dot Product (Inner Product):
$$\mathbf{u} \cdot \mathbf{v} = \sum_{i=1}^n u_i v_i = \|\mathbf{u}\| \|\mathbf{v}\| \cos(\theta)$$

### Cosine Similarity:
The foundation of modern Vector Databases (RAG search) and embedding comparisons:
$$\text{Cosine Similarity}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$$

* **$+1.0$:** Identical direction (high semantic similarity).
* **$0.0$:** Orthogonal (perpendicular, completely unrelated).
* **$-1.0$:** Exactly opposite direction.

```python
import numpy as np

u = np.array([1.0, 2.0, 3.0])
v = np.array([2.0, 4.0, 5.0])

dot_product = np.dot(u, v)
norm_u = np.linalg.norm(u)
norm_v = np.linalg.norm(v)

cosine_sim = dot_product / (norm_u * norm_v)
print(f"Cosine Similarity: {cosine_sim:.4f}")
```

---

## 3. Matrices & Matrix Multiplication

A matrix is a 2D rectangular array of numbers. Matrix multiplication transforms vectors from one dimensionality to another (exactly what every dense neural network layer does: $y = Wx + b$).

### Dimension Rule:
To multiply matrix $A$ by matrix $B$:
$$\text{Shape}(A) = (M \times K), \quad \text{Shape}(B) = (K \times N) \implies \text{Shape}(A \times B) = (M \times N)$$
*The inner dimensions ($K$) MUST match.*

```python
A = np.array([
    [1, 2],
    [3, 4],
    [5, 6]
]) # Shape: (3, 2)

B = np.array([
    [7, 8, 9],
    [10, 11, 12]
]) # Shape: (2, 3)

# Matrix Multiplication using the @ operator
C = A @ B  # Shape: (3, 3)
print("Resulting Matrix C:\n", C)
```

---

## 4. Tensors & Broadcasting Rules

A **tensor** generalizes scalars (0D), vectors (1D), and matrices (2D) to arbitrary dimensions:
* Image Tensor: `(Batch_Size, Channels, Height, Width)` $\to$ 4D Tensor.
* Text Embedding Tensor: `(Batch_Size, Sequence_Length, Embedding_Dim)` $\to$ 3D Tensor.

### Broadcasting Rules:
NumPy and PyTorch allow element-wise operations between arrays of different shapes if, trailing from right to left:
1. The dimensions are equal, **OR**
2. One of the dimensions is `1`.

```python
# Matrix of shape (3, 3)
M = np.ones((3, 3))

# Vector of shape (1, 3)
v = np.array([10, 20, 30])

# v is broadcast across all 3 rows automatically:
result = M + v
print("Broadcast Result:\n", result)
```

---

## Hands-On Deliverable
Navigate to [Week 3 Assignment](../assignments/Week3_Assignment.md) to implement pure Python matrix multiplication and compare execution speeds against vectorized NumPy.
