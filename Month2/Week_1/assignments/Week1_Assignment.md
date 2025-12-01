# Month 2, Week 1 Assignment: Simple Linear Regression from Scratch

## Objective

This assignment aims to solidify your understanding of Simple Linear Regression by implementing the algorithm from scratch using Python and NumPy. You will also implement the Mean Squared Error (MSE) and R-squared (R²) metrics to evaluate your model.

## Instructions

1.  **Create a new Python file:** Name it `simple_linear_regression.py`.

2.  **Generate Synthetic Data:**
    *   Create a synthetic dataset with a linear relationship.
    *   `X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])`
    *   `y = np.array([2, 4, 5, 4, 5, 7, 8, 9, 10, 12])`
    *   You can add some random noise to `y` for a more realistic scenario (optional).

3.  **Implement the `predict` Function:**
    *   Define a function `predict(X, b0, b1)` that takes the input features `X`, the intercept `b0`, and the slope `b1` as arguments.
    *   It should return the predicted `y` values based on the linear equation: `y_pred = b0 + b1 * X`.

4.  **Implement the Cost Function (MSE):**
    *   Define a function `mean_squared_error(y_true, y_pred)` that calculates the Mean Squared Error between the true `y` values and the predicted `y` values.
    *   `MSE = (1/n) * Σ (y_true - y_pred)²`

5.  **Implement R-squared (R²):**
    *   Define a function `r_squared(y_true, y_pred)` that calculates the R-squared value.
    *   `R² = 1 - (SSR_model / SST_total)`
        *   `SSR_model = Σ (y_true - y_pred)²` (This is `n * MSE`)
        *   `SST_total = Σ (y_true - y_mean)²` (where `y_mean` is the mean of `y_true`)

6.  **Implement Gradient Descent:**
    *   Define a function `gradient_descent(X, y, learning_rate, epochs)` that performs gradient descent to find the optimal `b0` and `b1`.
    *   **Initialization:** Start with `b0 = 0` and `b1 = 0`.
    *   **Loop for `epochs`:**
        *   Calculate `y_pred` using the current `b0` and `b1`.
        *   Calculate the gradients for `b0` and `b1`:
            *   `gradient_b0 = (-2/n) * Σ (y - y_pred)`
            *   `gradient_b1 = (-2/n) * Σ X * (y - y_pred)`
        *   Update `b0` and `b1`:
            *   `b0 = b0 - learning_rate * gradient_b0`
            *   `b1 = b1 - learning_rate * gradient_b1`
        *   (Optional) Print MSE every few epochs to observe convergence.
    *   Return the final `b0` and `b1`.

7.  **Main Execution Block:**
    *   Call `gradient_descent` with your synthetic data and chosen `learning_rate` (e.g., 0.01) and `epochs` (e.g., 1000).
    *   Use the returned `b0` and `b1` to make predictions on your `X` data.
    *   Calculate and print the final `MSE` and `R²` of your model.
    *   (Optional) Plot the original data points and your fitted regression line using `matplotlib`.

## Example Output (conceptual)

```
Optimal Intercept (b0): 1.5
Optimal Slope (b1): 0.9
Mean Squared Error (MSE): 0.8
R-squared (R²): 0.95
```

## Submission

*   Save your Python script as `simple_linear_regression.py`.
*   Ensure your code is well-commented and easy to understand.
