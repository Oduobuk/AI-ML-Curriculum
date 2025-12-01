# Month 2, Week 1 Exercises: Linear Regression

## Objective

These exercises are designed to reinforce your understanding of Linear Regression concepts, including residuals, cost functions, and evaluation metrics. You will perform some manual calculations and conceptual analysis.

---

### Exercise 1: Understanding Residuals and Best Fit Line

Consider the following small dataset:

| X | Y |
|---|---|
| 1 | 2 |
| 2 | 3 |
| 3 | 4 |
| 4 | 5 |

1.  **Perfect Fit:**
    *   What is the equation of the line that perfectly fits these data points? (i.e., `Y = β₀ + β₁X`)
    *   Calculate the residuals for each data point using this line.
    *   Calculate the Sum of Squared Residuals (SSR) for this line.

2.  **Imperfect Fit:**
    *   Now consider a different line: `Y = 1 + 0.5X`.
    *   Calculate the predicted Y values for each X using this line.
    *   Calculate the residuals for each data point using this line.
    *   Calculate the Sum of Squared Residuals (SSR) for this line.
    *   Which line is a better fit based on SSR?

---

### Exercise 2: Calculating MSE and RMSE

Using the "Imperfect Fit" line (`Y = 1 + 0.5X`) from Exercise 1 and the original data:

| X | Y |
|---|---|
| 1 | 2 |
| 2 | 3 |
| 3 | 4 |
| 4 | 5 |

1.  Calculate the Mean Squared Error (MSE).
2.  Calculate the Root Mean Squared Error (RMSE).

---

### Exercise 3: Calculating R-squared (R²)

Using the "Imperfect Fit" line (`Y = 1 + 0.5X`) from Exercise 1 and the original data:

| X | Y |
|---|---|
| 1 | 2 |
| 2 | 3 |
| 3 | 4 |
| 4 | 5 |

1.  Calculate the mean of the actual Y values (`y_mean`).
2.  Calculate the Total Sum of Squares (SST) around `y_mean`. (`SST = Σ (Y_actual - y_mean)²`)
3.  You already calculated the Sum of Squared Residuals (SSR_model) for the "Imperfect Fit" line in Exercise 1.
4.  Now, calculate the R-squared (R²) value for the "Imperfect Fit" line.
5.  Interpret the R² value you calculated. What does it tell you about the model?

---

### Exercise 4: Gradient Descent (Conceptual)

Imagine a very simple cost function `C(β) = (β - 5)²`. You want to find the `β` that minimizes this cost function using gradient descent.

1.  What is the derivative of this cost function with respect to `β`? (`∂C/∂β`)
2.  If you start with `β = 0` and a `learning_rate = 0.1`, what would be the value of `β` after the first update step?
3.  If you start with `β = 10` and a `learning_rate = 0.1`, what would be the value of `β` after the first update step?
4.  What do these two examples tell you about how gradient descent behaves when it's far from the minimum versus when it's closer?

---

### Exercise 5: True or False

Determine if the following statements are True or False. Justify your answer briefly.

1.  A higher R-squared value always means your linear regression model is good.
2.  In simple linear regression, the line of best fit always passes through the mean of X and the mean of Y.
3.  Polynomial regression is a non-linear model.
4.  Gradient Descent guarantees finding the global minimum for any cost function.
