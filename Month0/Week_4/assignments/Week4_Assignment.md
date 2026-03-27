Week 4: Assignment - Gradient Descent Optimization

📋 Project Overview

Implement gradient descent to find the minimum of a function and visualize the optimization process.

🎯 Part 1: Implement Gradient Descent (40%)

Write functions for:
1. Single-variable function and its derivative
2. Gradient descent algorithm with:
   - Learning rate parameter
   - Iteration limit
   - Convergence criterion
3. Track the optimization path

🎯 Part 2: Multiple Functions (30%)

Apply gradient descent to at least 3 different functions:
1. Quadratic: f(x) = (x - 3)² + 2
2. Quartic: f(x) = x⁴ - 3x² + 2
3. Multi-variable: f(x,y) = x² + y² - 4x

🎯 Part 3: Visualizations (20%)

Create plots showing:
1. Function curve with gradient descent path
2. Convergence plot (loss vs iteration)
3. Contour plot for 2D optimization

🎯 Part 4: Analysis (10%)

Write a report discussing:
1. How learning rate affects convergence
2. How starting point affects results
3. When gradient descent converges quickly vs slowly

💡 Implementation Tips

```python
def gradientdescent(f, df, x0, learningrate=0.01, max_iterations=1000):
    x = x0
    history = [x]
    
    for i in range(max_iterations):
        # Compute gradient
        grad = df(x)
        
        # Update x
        xnew = x - learningrate * grad
        
        # Check convergence
        if abs(x_new - x) < 1e-6:
            break
        
        x = x_new
        history.append(x)
    
    return x, history
```

📤 Submission

Submit:
1. Python script with implementation
2. Visualization plots
3. Analysis report

[Back to Lecture](../lecturenotes/Week4Math_Part2.md)
