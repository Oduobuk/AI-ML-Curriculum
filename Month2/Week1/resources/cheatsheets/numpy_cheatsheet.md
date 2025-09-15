# NumPy Cheat Sheet

## 1. Array Creation

```python
import numpy as np

# Create array from list
arr = np.array([1, 2, 3])

# Create array of zeros
zeros = np.zeros((3, 4))  # 3x4 array of zeros

# Create array of ones
ones = np.ones((2, 3))    # 2x3 array of ones

# Create identity matrix
identity = np.eye(3)      # 3x3 identity matrix

# Create array with range
range_arr = np.arange(0, 10, 2)  # array([0, 2, 4, 6, 8])

# Create linearly spaced array
lin_arr = np.linspace(0, 1, 5)  # array([0., 0.25, 0.5, 0.75, 1.])

# Create random array
rand_arr = np.random.rand(3, 2)  # 3x2 array of random numbers in [0,1)
```

## 2. Array Properties

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])

arr.shape    # (2, 3) - dimensions
arr.ndim     # 2 - number of dimensions
arr.size     # 6 - total number of elements
arr.dtype    # dtype('int64') - data type
arr.itemsize # 8 - size of each element in bytes
```

## 3. Array Indexing & Slicing

```python
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Indexing
arr[0, 1]    # 2 (0th row, 1st column)
arr[0][1]    # Same as above

# Slicing
arr[0:2, 1:3]  # Rows 0-1, columns 1-2
arr[:, 1]      # All rows, 1st column
arr[1, :]      # 1st row, all columns

# Boolean indexing
arr[arr > 5]   # array([6, 7, 8, 9])
```

## 4. Array Operations

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Element-wise operations
a + b    # array([5, 7, 9])
a * b    # array([4, 10, 18])
a ** 2   # array([1, 4, 9])

# Matrix multiplication
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
np.dot(A, B)  # or A @ B

# Aggregations
arr = np.array([[1, 2], [3, 4]])
arr.sum()         # 10
arr.mean()        # 2.5
arr.std()         # 1.118 (standard deviation)
arr.min()         # 1
arr.max()         # 4
arr.argmax()      # 3 (index of max element)
arr.cumsum(axis=0) # Column-wise cumulative sum
```

## 5. Array Manipulation

```python
# Reshaping
arr = np.arange(6)
arr.reshape(2, 3)  # Reshape to 2x3

# Transpose
arr.T  # or np.transpose(arr)

# Stacking
np.vstack((a, b))  # Vertical stack
np.hstack((a, b))  # Horizontal stack

# Splitting
np.hsplit(arr, 3)  # Split into 3 arrays horizontally
np.vsplit(arr, 2)  # Split into 2 arrays vertically
```

## 6. Linear Algebra

```python
# Matrix operations
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Determinant
np.linalg.det(A)

# Inverse
np.linalg.inv(A)

# Eigenvalues and eigenvectors
eigvals, eigvecs = np.linalg.eig(A)

# Solve linear equations
# Ax = b
x = np.linalg.solve(A, b)
```

## 7. Random Numbers

```python
# Random samples from uniform distribution
np.random.rand(3, 2)  # 3x2 array in [0,1)

# Random samples from normal distribution
np.random.randn(3, 2)  # Mean=0, std=1

# Random integers
np.random.randint(0, 10, size=(2, 3))  # 2x3 array of random ints in [0,10)

# Shuffle array
np.random.shuffle(arr)
```

## 8. Common Operations for Machine Learning

```python
# Feature scaling (standardization)
X = (X - X.mean(axis=0)) / X.std(axis=0)

# One-hot encoding
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
X_encoded = enc.fit_transform(X_categorical)

# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```

## 9. Performance Tips

- Use vectorized operations instead of loops
- Use `np.einsum` for complex tensor operations
- Pre-allocate arrays when possible
- Use `out` parameter to avoid creating temporary arrays
- For large arrays, consider using `np.memmap` for memory-mapped files
