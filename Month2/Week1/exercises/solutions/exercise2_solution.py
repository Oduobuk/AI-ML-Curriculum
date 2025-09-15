"""
Solution to Exercise 2: Multiple Linear Regression Implementation

This file contains the complete implementation of multiple linear regression
from scratch using NumPy with feature normalization.
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Sample dataset with multiple features
# X1: Size (sq ft), X2: Number of bedrooms
X = np.array([
    [2104, 3],
    [1600, 3],
    [2400, 3],
    [1416, 2],
    [3000, 4],
    [1985, 4],
    [1534, 3],
    [1427, 3],
    [1380, 3],
    [1494, 2]
])

# Target variable: House price (in $1000s)
Y = np.array([399, 329, 369, 232, 539, 299, 314, 198, 212, 242])

# Add a column of ones to X for the bias term (theta0)
X = np.column_stack((np.ones(len(X)), X))

# Initialize parameters
theta = np.zeros(X.shape[1])

# Hyperparameters
learning_rate = 0.01  # Adjusted learning rate for better convergence
iterations = 1000

def feature_normalize(X):
    """
    Normalize the features in X.
    
    Parameters:
    X (numpy.ndarray): Input features with shape (m, n+1)
    
    Returns:
    tuple: (X_norm, mu, sigma) where:
        - X_norm is the normalized version of X
        - mu is the mean of each feature
        - sigma is the standard deviation of each feature
    """
    # Don't normalize the first column (bias term)
    X_norm = X.copy()
    mu = np.mean(X[:, 1:], axis=0)
    sigma = np.std(X[:, 1:], axis=0, ddof=1)  # ddof=1 for sample standard deviation
    
    # Apply normalization to all features except the first column
    X_norm[:, 1:] = (X[:, 1:] - mu) / sigma
    
    return X_norm, mu, sigma

def hypothesis(X, theta):
    """
    Compute the hypothesis function for multiple linear regression.
    
    Parameters:
    X (numpy.ndarray): Input features with shape (m, n+1)
    theta (numpy.ndarray): Model parameters with shape (n+1,)
    
    Returns:
    numpy.ndarray: Predicted values with shape (m,)
    """
    return np.dot(X, theta)

def compute_cost(X, Y, theta):
    """
    Compute the cost function for linear regression.
    
    Parameters:
    X (numpy.ndarray): Input features with shape (m, n+1)
    Y (numpy.ndarray): Target values with shape (m,)
    theta (numpy.ndarray): Model parameters with shape (n+1,)
    
    Returns:
    float: The computed cost
    """
    m = len(Y)
    predictions = hypothesis(X, theta)
    squared_errors = (predictions - Y) ** 2
    return (1 / (2 * m)) * np.sum(squared_errors)

def gradient_descent(X, Y, theta, learning_rate, iterations):
    """
    Perform gradient descent to learn the parameters theta.
    
    Parameters:
    X (numpy.ndarray): Input features with shape (m, n+1)
    Y (numpy.ndarray): Target values with shape (m,)
    theta (numpy.ndarray): Initial model parameters with shape (n+1,)
    learning_rate (float): Learning rate for gradient descent
    iterations (int): Number of iterations to run gradient descent
    
    Returns:
    tuple: (final_theta, cost_history) where final_theta are the learned parameters
           and cost_history is a list of the cost at each iteration
    """
    m = len(Y)
    cost_history = []
    
    for _ in range(iterations):
        predictions = hypothesis(X, theta)
        errors = predictions - Y
        gradient = (1/m) * np.dot(X.T, errors)
        theta = theta - learning_rate * gradient
        cost = compute_cost(X, Y, theta)
        cost_history.append(cost)
    
    return theta, cost_history

def plot_cost_history(cost_history):
    """
    Plot the cost function over iterations.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(cost_history)
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title('Cost Function over Iterations')
    plt.show()

def plot_3d_data(X, Y, theta):
    """
    Create a 3D plot of the data points and the regression plane.
    """
    # Create a meshgrid for the plane
    x1 = np.linspace(X[:, 1].min(), X[:, 1].max(), 10)
    x2 = np.linspace(X[:, 2].min(), X[:, 2].max(), 10)
    x1_mesh, x2_mesh = np.meshgrid(x1, x2)
    
    # Calculate the predicted values for the plane
    y_pred = theta[0] + theta[1] * x1_mesh + theta[2] * x2_mesh
    
    # Create 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the data points
    ax.scatter(X[:, 1], X[:, 2], Y, c='b', marker='o', label='Training Data')
    
    # Plot the regression plane
    ax.plot_surface(x1_mesh, x2_mesh, y_pred, alpha=0.5, color='r', label='Regression Plane')
    
    ax.set_xlabel('Size (sq ft)')
    ax.set_ylabel('Number of Bedrooms')
    ax.set_zlabel('Price ($1000s)')
    ax.set_title('Multiple Linear Regression')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Print initial cost
    initial_cost = compute_cost(X, Y, theta)
    print(f"Initial cost: {initial_cost:.4f}")
    
    # Normalize features
    X_norm, mu, sigma = feature_normalize(X)
    
    # Run gradient descent
    theta, cost_history = gradient_descent(X_norm, Y, theta, learning_rate, iterations)
    
    # Print final parameters and cost
    final_cost = compute_cost(X_norm, Y, theta)
    print(f"Final parameters: {theta}")
    print(f"Final cost: {final_cost:.4f}")
    
    # Plot cost history
    plot_cost_history(cost_history)
    
    # Plot 3D data with regression plane
    plot_3d_data(X_norm, Y, theta)
    
    # Make a prediction for a new house
    # Size: 1650 sq ft, 3 bedrooms
    x_new = np.array([1, 1650, 3])
    # Normalize the new example using the same mu and sigma
    x_new_norm = (x_new - np.concatenate(([0], mu))) / np.concatenate(([1], sigma))
    prediction = np.dot(x_new_norm, theta)
    print(f"Predicted price for a 1650 sq ft, 3 br house: ${prediction*1000:.2f}")
    
    # Print the equation of the model
    print("\nFinal Model:")
    print(f"Price = {theta[0]:.2f} + {theta[1]:.2f} * (size - {mu[0]:.2f})/{sigma[0]:.2f} + {theta[2]:.2f} * (bedrooms - {mu[1]:.2f})/{sigma[1]:.2f}")
