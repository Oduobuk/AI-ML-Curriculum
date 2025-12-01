# Month 2, Week 1 Cheatsheet: Linear Regression Fundamentals

## 1. Core Concepts

*   **Regression:** A supervised learning task to predict a **continuous numerical output** (dependent variable `Y`) based on one or more **input features** (independent variables `X`).
*   **Linear Model:** Assumes a linear relationship between inputs and output.

---

## 2. Types of Linear Regression

### a) Simple Linear Regression (SLR)
*   **Equation:** `Y = β₀ + β₁X + ε`
    *   `Y`: Dependent variable
    *   `X`: Independent variable
    *   `β₀`: Y-intercept
    *   `β₁`: Slope (coefficient of X)
    *   `ε`: Error term (residual)

### b) Multiple Linear Regression (MLR)
*   **Equation:** `Y = β₀ + β₁X₁ + β₂X₂ + ... + βₚXₚ + ε`
    *   `X₁, ..., Xₚ`: Multiple independent variables
    *   `β₁, ..., βₚ`: Coefficients for each independent variable

### c) Polynomial Regression
*   **Equation:** `Y = β₀ + β₁X + β₂X² + ... + βₚXᵖ + ε`
    *   Models non-linear relationships using polynomial terms of the independent variable. Still considered "linear" because it's linear in the coefficients `β`.

---

## 3. Model Fitting: Least Squares Method

*   **Goal:** Find the `β₀` and `β₁` (or `β`s) that best fit the data.
*   **Residual (Error):** `eᵢ = Yᵢ_actual - Yᵢ_predicted`
    *   The vertical distance between an observed data point and the regression line.
*   **Sum of Squared Residuals (SSR):** `SSR = Σ (Yᵢ_actual - Yᵢ_predicted)²`
    *   The sum of the squares of all residuals.
    *   The **Least Squares Method** minimizes this `SSR` to find the optimal regression line.

---

## 4. Optimization: Gradient Descent

*   **Cost Function (Loss Function):** A function that quantifies the error of our model's predictions. For linear regression, `MSE` (Mean Squared Error) is commonly used.
    *   `MSE = (1/n) * Σ (Yᵢ_actual - Yᵢ_predicted)²`
*   **Gradient Descent:** An iterative optimization algorithm to find the parameters (`β`s) that minimize the cost function.
    1.  **Initialize:** Start with random `β` values.
    2.  **Calculate Gradient:** Compute the partial derivative of the cost function with respect to each `β`. This indicates the direction of steepest ascent.
    3.  **Update Parameters:** Adjust `β`s in the opposite direction of the gradient (downhill) by a step size determined by the `learning_rate`.
        *   `β_new = β_old - learning_rate * (∂Cost/∂β_old)`
    4.  **Repeat:** Iterate until convergence (gradients are near zero, or maximum epochs reached).

---

## 5. Evaluation Metrics

*   **Mean Squared Error (MSE):** `MSE = (1/n) * Σ (Yᵢ_actual - Yᵢ_predicted)²`
    *   Average of squared errors. Penalizes larger errors more.
*   **Root Mean Squared Error (RMSE):** `RMSE = √MSE`
    *   Error in the same units as `Y`, making it more interpretable.
*   **R-squared (R²):** `R² = 1 - (SSR_model / SST_total)`
    *   `SSR_model`: Sum of squared residuals of the model.
    *   `SST_total`: Total sum of squares (variance of `Y` around its mean).
    *   Measures the proportion of variance in `Y` explained by `X`. Ranges from 0 to 1. Higher is better.
*   **Adjusted R-squared:** `Adjusted R² = 1 - [(1 - R²) * (n - 1) / (n - p - 1)]`
    *   `n`: number of samples, `p`: number of predictors.
    *   Adjusts R² for the number of predictors, penalizing models with too many unnecessary features. Useful for comparing models with different numbers of independent variables.

---

## 6. Assumptions of Linear Regression

*   **Linearity:** The relationship between X and Y is linear.
*   **Independence:** Observations are independent of each other.
*   **Homoscedasticity:** The variance of residuals is constant across all levels of X.
*   **Normality:** Residuals are normally distributed.
*   **No Multicollinearity:** Independent variables are not highly correlated with each other (for MLR).

---

## 7. Overfitting

*   Occurs when a model learns the training data too well, including noise, leading to poor performance on unseen data.
*   Can be a concern with polynomial regression if the degree is too high.
