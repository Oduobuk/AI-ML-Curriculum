# Linear Regression Cheat Sheet

## 1. Simple Linear Regression

### Model
$$
h_\theta(x) = \theta_0 + \theta_1 x
$$

### Cost Function (Mean Squared Error)
$$
J(\theta_0, \theta_1) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2
$$

### Gradient Descent Update Rules
Repeat until convergence:
$$
\theta_0 := \theta_0 - \alpha \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})
$$
$$
\theta_1 := \theta_1 - \alpha \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) \cdot x^{(i)}
$$

## 2. Multiple Linear Regression

### Model
$$
h_\theta(x) = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + ... + \theta_n x_n
$$

### Vectorized Form
$$
h_\theta(X) = X\theta
$$

### Cost Function
$$
J(\theta) = \frac{1}{2m} (X\theta - y)^T (X\theta - y)
$$

### Gradient Descent (Vectorized)
Repeat until convergence:
$$
\theta := \theta - \alpha \frac{1}{m} X^T (X\theta - y)
$$

## 3. Normal Equation
$$
\theta = (X^T X)^{-1} X^T y
$$

## 4. Evaluation Metrics

### R-squared (Coefficient of Determination)
$$
R^2 = 1 - \frac{SS_{res}}{SS_{tot}} = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}
$$

### Mean Squared Error (MSE)
$$
MSE = \frac{1}{m} \sum_{i=1}^{m} (y_i - \hat{y}_i)^2
$$

### Root Mean Squared Error (RMSE)
$$
RMSE = \sqrt{MSE}
$$

## 5. Feature Scaling

### Mean Normalization
$$
x' = \frac{x - \mu}{max - min}
$$

### Standardization (Z-score Normalization)
$$
x' = \frac{x - \mu}{\sigma}
$$

## 6. Regularization

### Ridge Regression (L2)
$$
J(\theta) = \frac{1}{2m} \left[ \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2 + \lambda \sum_{j=1}^{n} \theta_j^2 \right]
$$

### Lasso Regression (L1)
$$
J(\theta) = \frac{1}{2m} \left[ \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2 + \lambda \sum_{j=1}^{n} |\theta_j| \right]
$$

## 7. Common Issues & Solutions

| Issue | Possible Solution |
|-------|-------------------|
| High Bias | Add polynomial features, get more features |
| High Variance | Get more training data, reduce features, increase λ |
| Non-linear data | Add polynomial features, use different model |
| Multicollinearity | Remove correlated features, use regularization |
