Week 3: Essential Math for AI/ML (Part 1)

Learning Objectives

By the end of this week, you will be able to:
- [ ] Understand algebraic concepts (exponents, logarithms, equations)
- [ ] Graph and analyze linear functions
- [ ] Calculate basic statistics (mean, median, mode, std dev)
- [ ] Understand probability basics
- [ ] Use NumPy and Matplotlib for mathematical computations and visualization

1. Mathematical Foundations

1.1 Exponents (Powers)

Exponents show how many times to multiply a number by itself.

```
2³ = 2 × 2 × 2 = 8
```

In Python:
```python
2  3           # 8
10  2          # 100
5  0           # 1 (anything to power 0 is 1)
2  -1          # 0.5 (negative powers are reciprocals)
```

Why it matters for ML: Neural networks use exponentials; optimization uses powers.

1.2 Logarithms (Inverse of Exponents)

Logarithms answer the question: "What power do I raise this to get that number?"

```
log₂(8) = 3   (because 2³ = 8)
log₁₀(100) = 2   (because 10² = 100)
```

In Python:
```python
import math

math.log(8, 2)      # 3.0 (log base 2 of 8)
math.log10(100)     # 2.0 (log base 10 of 100)
math.log(2.718)     # 1.0 (natural log)
```

Why it matters for ML: Loss functions use logarithms; exponentials and logs are inverse operations.

1.3 Linear Equations

A linear equation is a relationship between two variables that graphs as a straight line.

```
y = mx + b

where:
m = slope (steepness)
b = y-intercept (where line crosses y-axis)
```

Example:
```python
y = 2x + 1
For each x value, calculate y
def linear(x):
    return 2 * x + 1

linear(0)     # 1 (y-intercept)
linear(1)     # 3
linear(2)     # 5
linear(3)     # 7
```

Solving for x:
```python
If y = 2x + 1, what's x when y = 7?
7 = 2x + 1
6 = 2x
3 = x

In Python:
def solveforx(y):
    return (y - 1) / 2

solveforx(7)  # 3.0
```

1.4 Quadratic Equations

Quadratic equations have a squared term and graph as parabolas.

```
y = ax² + bx + c
```

Example:
```python
def quadratic(x):
    return x2 - 4*x + 3

quadratic(0)     # 3
quadratic(1)     # 0
quadratic(2)     # -1
quadratic(3)     # 0
```

2. Statistics: Understanding Data

2.1 Mean (Average)

Mean is the sum of all values divided by count.

```python
scores = [80, 85, 90, 95, 100]
mean = sum(scores) / len(scores)
print(mean)  # 90
```

In NumPy:
```python
import numpy as np

scores = np.array([80, 85, 90, 95, 100])
mean = np.mean(scores)  # 90
```

2.2 Median (Middle Value)

Median is the middle value when sorted.

```python
scores = [80, 85, 90, 95, 100]
Already sorted, middle value (index 2) is 90
median = scores[len(scores)//2]  # 90

Using NumPy
median = np.median(scores)  # 90

If even number of values, average the two middle
scores2 = [80, 85, 90, 95]
Median = (85 + 90) / 2 = 87.5
```

2.3 Mode (Most Frequent Value)

Mode is the value that appears most often.

```python
from statistics import mode

scores = [80, 85, 85, 90, 95]
mode_value = mode(scores)  # 85 (appears twice)
```

2.4 Standard Deviation (Spread)

Standard Deviation shows how spread out the data is.

- Low std dev = values are close to mean
- High std dev = values are far from mean

```python
import numpy as np

scores1 = np.array([80, 85, 90, 95, 100])  # Spread out
std1 = np.std(scores1)  # ~7.07

scores2 = np.array([89, 90, 90, 90, 91])   # Close together
std2 = np.std(scores2)  # ~0.45
```

Calculating std dev (conceptually):
1. Find mean
2. Find distance from each value to mean
3. Square those distances
4. Find average of squared distances
5. Take square root

2.5 Variance

Variance is std dev squared. It's also a measure of spread.

```python
variance = np.var(scores)  # Standard deviation²
std_dev = np.sqrt(variance)
```

3. Probability: Basics

Probability measures how likely something is to happen.

```
Probability = (Number of favorable outcomes) / (Total number of outcomes)
```

3.1 Basic Probability

```python
Coin flip
Probability of heads = 1/2 = 0.5

Six-sided die
Probability of rolling 3 = 1/6 ≈ 0.167

Drawing from deck of cards
Probability of drawing an ace = 4/52 ≈ 0.077
```

3.2 Independent vs. Dependent Events

Independent: One event doesn't affect the other
```python
Flip coin, then roll die
P(heads AND 3) = P(heads) × P(3) = 0.5 × (1/6) ≈ 0.083
prob_heads = 1/2
prob_three = 1/6
probboth = probheads * prob_three
```

Dependent: One event affects the other
```python
Draw card, don't replace, draw again
First card probability = 52/52
Second card probability = 51/51 (one less card)
```

3.3 Conditional Probability

Conditional probability = Probability of event A given that B happened

```python
P(B|A) = "Probability of B given A"

Example: Medical test
P(disease | positive test) = ?
This is: among people with positive tests, 
what fraction actually has the disease?
```

4. Visualizing Data with Matplotlib

4.1 Basic Plotting

```python
import matplotlib.pyplot as plt

Simple line plot
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

plt.plot(x, y)
plt.xlabel("X Values")
plt.ylabel("Y Values")
plt.title("Simple Line Plot")
plt.show()
```

4.2 Different Plot Types

Scatter plot:
```python
import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5]
y = [2, 4, 5, 8, 10]

plt.scatter(x, y)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Scatter Plot")
plt.show()
```

Histogram (distribution):
```python
scores = [80, 85, 85, 90, 90, 90, 95, 100]

plt.hist(scores, bins=5)
plt.xlabel("Score")
plt.ylabel("Frequency")
plt.title("Score Distribution")
plt.show()
```

Bar plot:
```python
categories = ["Python", "Java", "C++", "JavaScript"]
popularity = [85, 60, 45, 70]

plt.bar(categories, popularity)
plt.ylabel("Popularity (%)")
plt.title("Programming Language Popularity")
plt.show()
```

4.3 Multiple Plots

```python
import matplotlib.pyplot as plt

Create 2x2 grid of subplots
fig, axes = plt.subplots(2, 2)

axes[0, 0].plot([1, 2, 3], [1, 4, 9])
axes[0, 0].set_title("Line Plot")

axes[0, 1].scatter([1, 2, 3], [1, 4, 9])
axes[0, 1].set_title("Scatter")

axes[1, 0].bar(["A", "B", "C"], [10, 20, 15])
axes[1, 0].set_title("Bar Chart")

axes[1, 1].hist([1, 1, 2, 2, 2, 3, 3, 3, 3])
axes[1, 1].set_title("Histogram")

plt.tight_layout()
plt.show()
```

5. Introduction to NumPy

NumPy is Python's library for numerical computing.

5.1 NumPy Arrays

```python
import numpy as np

Create arrays
arr1 = np.array([1, 2, 3, 4, 5])
arr2 = np.array([10, 20, 30])

Arrays of zeros/ones
zeros = np.zeros(5)      # [0, 0, 0, 0, 0]
ones = np.ones(5)        # [1, 1, 1, 1, 1]
range_arr = np.arange(0, 10, 2)  # [0, 2, 4, 6, 8]

2D arrays (matrices)
matrix = np.array([[1, 2, 3], 
                   [4, 5, 6]])

Shape and size
arr = np.array([1, 2, 3, 4, 5])
arr.shape      # (5,)
arr.size       # 5
arr.dtype      # dtype('int64')
```

5.2 Array Operations

```python
import numpy as np

arr1 = np.array([1, 2, 3, 4])
arr2 = np.array([10, 20, 30, 40])

Element-wise operations
arr1 + arr2      # [11, 22, 33, 44]
arr1 * arr2      # [10, 40, 90, 160]
arr1  2        # [1, 4, 9, 16]
np.sqrt(arr1)    # [1, 1.41, 1.73, 2]

Scalar operations
arr1 + 5         # [6, 7, 8, 9]
arr1 * 2         # [2, 4, 6, 8]
```

5.3 Statistical Functions

```python
import numpy as np

data = np.array([80, 85, 90, 95, 100])

np.mean(data)      # 90
np.median(data)    # 90
np.std(data)       # ~7.07
np.var(data)       # 50
np.min(data)       # 80
np.max(data)       # 100
np.sum(data)       # 450
```

6. Putting It Together: Analysis Project

```python
import numpy as np
import matplotlib.pyplot as plt

Sample test scores
scores = np.array([65, 70, 75, 80, 85, 85, 90, 95, 100])

Calculate statistics
mean_score = np.mean(scores)
median_score = np.median(scores)
std_score = np.std(scores)

print(f"Mean: {mean_score}")
print(f"Median: {median_score}")
print(f"Std Dev: {std_score:.2f}")

Visualize
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.hist(scores, bins=5, edgecolor='black')
plt.axvline(meanscore, color='r', linestyle='--', label=f'Mean: {meanscore:.1f}')
plt.xlabel("Score")
plt.ylabel("Frequency")
plt.title("Score Distribution")
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(range(len(scores)), scores)
plt.plot(range(len(scores)), scores, alpha=0.5)
plt.ylabel("Score")
plt.title("Scores Over Time")

plt.tight_layout()
plt.show()
```

🎯 Key Concepts

| Concept | Definition | ML Use |
|---------|-----------|--------|
| Exponents | Repeated multiplication | Neural network activations |
| Logarithms | Inverse of exponents | Loss functions, probability |
| Linear function | Straight-line relationship | Regression models |
| Mean | Average | Center of data |
| Std Deviation | Spread around mean | Data normalization |
| Probability | Likelihood of event | Classification models |

📚 Additional Resources

- [Khan Academy: Algebra](https://www.khanacademy.org/math/algebra1)
- [Khan Academy: Statistics](https://www.khanacademy.org/math/statistics-probability)
- [NumPy Documentation](https://numpy.org/doc/)
- [Matplotlib Documentation](https://matplotlib.org/)

Ready for exercises? → [Week 3 Exercises](../exercises/Week3_Exercises.md)
