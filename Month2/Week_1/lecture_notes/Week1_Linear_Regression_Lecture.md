# Month 2, Week 1: Linear Regression

## 1. Introduction to Regression

**Regression analysis** is a fundamental supervised learning technique used to model the relationship between a dependent variable (the target or outcome) and one or more independent variables (the predictors or features). Its primary goals include:
*   **Prediction/Forecasting:** Estimating the value of the dependent variable for new, unseen data.
*   **Relationship Analysis:** Understanding the strength and nature of the relationship between variables.
*   **Trend Evaluation:** Identifying trends and making future estimates.

## 2. Simple Linear Regression

**Simple Linear Regression (SLR)** models the relationship between a single independent variable (X) and a single dependent variable (Y) using a straight line. The equation of this line is:

`Y = β₀ + β₁X + ε`

Where:
*   `Y`: The dependent variable.
*   `X`: The independent variable.
*   `β₀`: The Y-intercept (the value of Y when X is 0).
*   `β₁`: The slope of the line (the change in Y for a one-unit change in X).
*   `ε`: The error term (residuals), representing the difference between the observed Y and the predicted Y.

The goal of linear regression is to find the optimal values for `β₀` and `β₁` that best fit the data.

## 3. Multiple Linear Regression

**Multiple Linear Regression (MLR)** extends SLR to cases with two or more independent variables. The equation becomes:

`Y = β₀ + β₁X₁ + β₂X₂ + ... + βₚXₚ + ε`

Where:
*   `X₁, X₂, ..., Xₚ`: Multiple independent variables.
*   `β₁, β₂, ..., βₚ`: The coefficients for each independent variable.

## 4. The Least Squares Method

The "best fit" line in linear regression is typically determined using the **Ordinary Least Squares (OLS)** method. OLS aims to minimize the **Sum of Squared Residuals (SSR)**.

*   **Residuals (Errors):** For each data point, a residual is the vertical distance between the observed value (`Y_actual`) and the value predicted by the regression line (`Y_predicted`).
    `Residual = Y_actual - Y_predicted`
*   **Sum of Squared Residuals (SSR):** The sum of the squares of all residuals. Squaring ensures that positive and negative errors do not cancel out and penalizes larger errors more heavily.
    `SSR = Σ (Y_actual - Y_predicted)²`

The line that minimizes this sum is considered the best-fitting line.

## 5. Cost Function and Gradient Descent

*   **Cost Function (Loss Function):** In machine learning, the SSR (or often the Mean Squared Error, MSE) serves as the **cost function**. The objective of the learning algorithm is to find the model parameters (`β₀`, `β₁`, etc.) that minimize this cost function.
*   **Gradient Descent:** An iterative optimization algorithm used to find the minimum of a function (our cost function).
    1.  **Initialize Parameters:** Start with random initial values for `β₀` and `β₁`.
    2.  **Calculate Gradient:** Compute the gradient (partial derivatives) of the cost function with respect to each parameter. The gradient indicates the direction of the steepest ascent.
    3.  **Update Parameters:** Adjust the parameters in the opposite direction of the gradient (downhill) by a small amount determined by the **learning rate**.
        `New Parameter = Old Parameter - (Learning Rate * Gradient)`
    4.  **Repeat:** Continue steps 2 and 3 until the parameters converge (i.e., the step size becomes very small, indicating the minimum has been reached).

## 6. Evaluation Metrics for Regression

To assess how well a linear regression model performs, several metrics are commonly used:

*   **Mean Squared Error (MSE):** The average of the squared residuals. It provides a measure of the average magnitude of the errors.
    `MSE = (1/n) * Σ (Y_actual - Y_predicted)²`
*   **Root Mean Squared Error (RMSE):** The square root of the MSE. It is often preferred over MSE because it is in the same units as the dependent variable, making it more interpretable.
    `RMSE = √MSE`
*   **R-squared (R²):** Also known as the coefficient of determination. It measures the proportion of the variance in the dependent variable that can be explained by the independent variable(s) in the model.
    *   `R² = 1 - (SSR_model / SST_total)`
    *   `SSR_model`: Sum of squared residuals of the regression model.
    *   `SST_total`: Total sum of squares (variance of the dependent variable around its mean).
    *   R² ranges from 0 to 1. A higher R² indicates a better fit.
*   **Adjusted R-squared:** A modified version of R² that adjusts for the number of predictors in the model. It increases only if the new predictor improves the model more than would be expected by chance, making it useful for comparing models with different numbers of independent variables.

## 7. Polynomial Regression

**Polynomial Regression** is a form of linear regression where the relationship between the independent variable `X` and the dependent variable `Y` is modeled as an nth degree polynomial. While it models non-linear relationships, it is still considered a form of linear regression because it is linear in its parameters (`β`).

`Y = β₀ + β₁X + β₂X² + ... + βₚXᵖ + ε`

This allows the model to fit a wider range of curves to the data, capturing more complex relationships than a simple straight line.

## 8. Overfitting (Brief Mention)

**Overfitting** occurs when a model learns the training data too well, including its noise and outliers, leading to poor performance on new, unseen data. In regression, this can happen if a model (especially polynomial regression with a high degree) is too complex for the underlying data, fitting every wiggle and noise point in the training set.

---
## Recommended Textbooks

*   **Introduction to Statistical Learning (ISLR)** – Chapter 3
*   **Hands-On Machine Learning** (Aurélien Géron) – Regression chapter
*   **Pattern Recognition & Machine Learning** (Bishop) – Linear Models
