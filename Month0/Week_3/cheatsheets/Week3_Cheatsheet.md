Week 3: Essential Math Part 1 Cheatsheet

Quick reference for statistics, probability, and visualization.

📊 Statistics

Mean, Median, Mode
```python
import numpy as np
from statistics import mode

data = [80, 85, 90, 95, 100]

Mean (average)
mean = np.mean(data)  # 90
mean_manual = sum(data) / len(data)

Median (middle value)
median = np.median(data)  # 90

Mode (most frequent)
mode_val = mode(data)
```

Std Dev & Variance
```python
std_dev = np.std(data)    # Spread
variance = np.var(data)   # Std dev squared
```

Min, Max, Range
```python
min_val = np.min(data)    # 80
max_val = np.max(data)    # 100
rangeval = maxval - min_val  # 20
```

🎲 Probability

Basic Formulas
```
P(A) = Favorable Outcomes / Total Outcomes

P(A and B) = P(A) × P(B)  [Independent]
P(A or B) = P(A) + P(B) - P(A and B)
P(A | B) = P(A and B) / P(B)  [Conditional]
```

Examples
```python
Coin flip: P(heads) = 1/2
prob_heads = 1/2

Dice: P(6) = 1/6
prob_six = 1/6

Two dice: P(both 6) = 1/36
probbothsix = (1/6) * (1/6)
```

📈 Matplotlib Plotting

Basic Line Plot
```python
import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5]
y = [1, 4, 9, 16, 25]

plt.plot(x, y)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Title")
plt.show()
```

Scatter Plot
```python
plt.scatter(x, y)
plt.show()
```

Histogram
```python
data = [65, 70, 75, 80, 85, 90, 95]

plt.hist(data, bins=5)
plt.xlabel("Score")
plt.ylabel("Frequency")
plt.show()
```

Bar Chart
```python
categories = ["A", "B", "C"]
values = [10, 20, 15]

plt.bar(categories, values)
plt.show()
```

Multiple Subplots
```python
fig, axes = plt.subplots(2, 2)

axes[0, 0].plot(x, y)
axes[0, 1].scatter(x, y)
axes[1, 0].bar(x, y)
axes[1, 1].hist(y)

plt.tight_layout()
plt.show()
```

Customization
```python
plt.figure(figsize=(10, 6))
plt.plot(x, y, 'r-', linewidth=2, label='Data')
plt.plot(x, y2, 'b--', label='Comparison')
plt.legend()
plt.grid(True)
plt.show()
```

🔢 NumPy Functions

Array Creation
```python
import numpy as np

np.array([1, 2, 3])        # From list
np.zeros(5)                # [0, 0, 0, 0, 0]
np.ones(5)                 # [1, 1, 1, 1, 1]
np.arange(0, 10, 2)        # [0, 2, 4, 6, 8]
np.linspace(0, 1, 5)       # 5 points from 0 to 1
```

Statistical Functions
```python
data = np.array([1, 2, 3, 4, 5])

np.mean(data)        # Average
np.median(data)      # Middle value
np.std(data)         # Standard deviation
np.var(data)         # Variance
np.min(data)         # Minimum
np.max(data)         # Maximum
np.sum(data)         # Sum
np.sort(data)        # Sorted
```

Element-Wise Operations
```python
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])

arr1 + arr2          # [5, 7, 9]
arr1 * arr2          # [4, 10, 18]
np.sqrt(arr1)        # [1, 1.41, 1.73]
```

📐 Common Statistical Patterns

Percentile
```python
data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
p75 = np.percentile(data, 75)  # 75th percentile
```

Quartiles
```python
q1 = np.percentile(data, 25)
q2 = np.percentile(data, 50)  # Median
q3 = np.percentile(data, 75)
```

Standard Error
```python
std_error = np.std(data) / np.sqrt(len(data))
```

Covariance
```python
x = np.array([1, 2, 3])
y = np.array([2, 4, 6])
covar = np.cov(x, y)
```

[Back to Lecture](../lecturenotes/Week3Math_Part1.md)
