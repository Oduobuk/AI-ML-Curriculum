"""
Polynomial Regression and the Bias-Variance Tradeoff
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, learning_curve
import seaborn as sns

# Set style for plots
sns.set(style='whitegrid')

# 1. Generate Sample Data
def generate_sample_data(n_samples=100, noise=0.5):
    """Generate non-linear sample data."""
    np.random.seed(42)
    X = 6 * np.random.rand(n_samples, 1) - 3  # X between -3 and 3
    y = 0.5 * X**2 + X + 2 + np.random.randn(n_samples, 1) * noise
    return X, y

# 2. Plot Learning Curves
def plot_learning_curve(model, X, y, title):
    """Plot learning curves for a model."""
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=5, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='neg_mean_squared_error'
    )
    
    train_scores_mean = -np.mean(train_scores, axis=1)
    test_scores_mean = -np.mean(test_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training error')
    plt.plot(train_sizes, test_scores_mean, 'o-', color='g', label='Validation error')
    plt.xlabel('Training examples')
    plt.ylabel('Mean Squared Error')
    plt.title(f'Learning Curves - {title}')
    plt.legend()
    plt.grid(True)
    plt.show()

# 3. Main Function
def main():
    # Generate data
    X, y = generate_sample_data(100, 0.5)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3.1 Linear Regression
    print("Fitting Linear Regression...")
    linear_reg = LinearRegression()
    linear_reg.fit(X_train, y_train)
    
    # 3.2 Polynomial Regression
    print("Fitting Polynomial Regression...")
    polynomial_reg = Pipeline([
        ('poly_features', PolynomialFeatures(degree=2, include_bias=False)),
        ('std_scaler', StandardScaler()),
        ('lin_reg', LinearRegression())
    ])
    polynomial_reg.fit(X_train, y_train)
    
    # 3.3 Ridge Regression (L2 Regularization)
    print("Fitting Ridge Regression...")
    ridge_reg = Pipeline([
        ('poly_features', PolynomialFeatures(degree=10, include_bias=False)),
        ('std_scaler', StandardScaler()),
        ('ridge', Ridge(alpha=1, solver='cholesky'))
    ])
    ridge_reg.fit(X_train, y_train)
    
    # 3.4 Lasso Regression (L1 Regularization)
    print("Fitting Lasso Regression...")
    lasso_reg = Pipeline([
        ('poly_features', PolynomialFeatures(degree=10, include_bias=False)),
        ('std_scaler', 'passthrough'),
        ('lasso', Lasso(alpha=0.1, max_iter=1000))
    ])
    lasso_reg.fit(X_train, y_train)
    
    # 4. Plot Results
    plt.figure(figsize=(12, 8))
    
    # Plot training data
    plt.scatter(X_train, y_train, color='blue', label='Training data')
    
    # Sort X values for plotting
    X_plot = np.linspace(-3, 3, 100).reshape(-1, 1)
    
    # Plot Linear Regression
    y_linear = linear_reg.predict(X_plot)
    plt.plot(X_plot, y_linear, 'r-', linewidth=2, label='Linear Regression')
    
    # Plot Polynomial Regression
    y_poly = polynomial_reg.predict(X_plot)
    plt.plot(X_plot, y_poly, 'g-', linewidth=2, label='Polynomial (deg=2)')
    
    # Plot Ridge Regression
    y_ridge = ridge_reg.predict(X_plot)
    plt.plot(X_plot, y_ridge, 'y-', linewidth=2, label='Ridge (L2)')
    
    # Plot Lasso Regression
    y_lasso = lasso_reg.predict(X_plot)
    plt.plot(X_plot, y_lasso, 'm-', linewidth=2, label='Lasso (L1)')
    
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Polynomial Regression with Regularization')
    plt.legend()
    plt.show()
    
    # 5. Model Evaluation
    models = {
        'Linear': linear_reg,
        'Polynomial (deg=2)': polynomial_reg,
        'Ridge (L2)': ridge_reg,
        'Lasso (L1)': lasso_reg
    }
    
    print("\nModel Performance (MSE):")
    for name, model in models.items():
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print(f"{name}: {mse:.4f}")
    
    # 6. Plot Learning Curves
    print("\nGenerating Learning Curves...")
    for name, model in models.items():
        plot_learning_curve(model, X, y, name)

if __name__ == "__main__":
    main()
