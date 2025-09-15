"""
Solution to Exercise 1: Simple Linear Regression Implementation

This file contains the complete implementation of simple linear regression
from scratch using NumPy.
"""
import numpy as np
import matplotlib.pyplot as plt

# Sample dataset
X = np.array([1, 2, 3, 4, 5])  # Feature
Y = np.array([2, 4, 5, 4, 5])   # Target

# Add a column of ones to X for the bias term (theta0)
X = np.column_stack((np.ones_like(X), X))

# Initialize parameters
theta = np.zeros(2)  # [theta0, theta1]

# Hyperparameters
learning_rate = 0.01
iterations = 1000

def hypothesis(X, theta):
    """
    Compute the hypothesis function for linear regression.
    
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

def plot_results(X, Y, theta, cost_history):
    """
    Plot the data points, the regression line, and the cost history.
    """
    plt.figure(figsize=(15, 5))
    
    # Plot data and regression line
    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 1], Y, color='blue', label='Training Data')
    plt.plot(X[:, 1], X @ theta, color='red', label='Linear Regression')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Linear Regression Fit')
    plt.legend()
    
    # Plot cost history
    plt.subplot(1, 2, 2)
    plt.plot(cost_history)
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title('Cost Function over Iterations')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Print initial cost
    initial_cost = compute_cost(X, Y, theta)
    print(f"Initial cost: {initial_cost:.4f}")
    
    # Run gradient descent
    theta, cost_history = gradient_descent(X, Y, theta, learning_rate, iterations)
    
    # Print final parameters and cost
    final_cost = compute_cost(X, Y, theta)
    print(f"Final parameters: {theta}")
    print(f"Final cost: {final_cost:.4f}")
    
    # Plot results
    plot_results(X, Y, theta, cost_history)
    
    # Make a prediction for x = 6
    x_new = np.array([1, 6])  # Don't forget the bias term
    prediction = np.dot(x_new, theta)
    print(f"Prediction for x=6: {prediction:.4f}")
