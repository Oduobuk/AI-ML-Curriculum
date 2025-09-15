"""
Solution to Exercise 3: Gradient Descent Implementation

This file contains the complete implementation of different gradient descent variants
(Batch, Stochastic, and Mini-batch) for linear regression.
"""
import numpy as np
import matplotlib.pyplot as plt
import time

# Set random seed for reproducibility
np.random.seed(42)

# Generate sample data
m = 1000  # Number of training examples
X = 2 * np.random.rand(m, 1)  # Random values between 0 and 2
y = 4 + 3 * X + np.random.randn(m, 1)  # y = 4 + 3x + noise

# Add bias term to X
X_b = np.c_[np.ones((m, 1)), X]  # Add x0 = 1 to each instance

# Initialize parameters
theta = np.random.randn(2, 1)  # Random initialization

# Hyperparameters
learning_rate = 0.1
n_epochs = 50
batch_size = 100  # For mini-batch gradient descent

def compute_cost(X, y, theta):
    """
    Compute the mean squared error cost.
    
    Parameters:
    X (numpy.ndarray): Input features with shape (m, n+1)
    y (numpy.ndarray): Target values with shape (m, 1)
    theta (numpy.ndarray): Model parameters with shape (n+1, 1)
    
    Returns:
    float: The computed cost
    """
    m = len(y)
    predictions = X.dot(theta)
    errors = predictions - y
    return (1/(2*m)) * np.sum(errors**2)

def batch_gradient_descent(X, y, theta, learning_rate=0.1, n_epochs=100):
    """
    Perform batch gradient descent.
    
    Parameters:
    X (numpy.ndarray): Input features with shape (m, n+1)
    y (numpy.ndarray): Target values with shape (m, 1)
    theta (numpy.ndarray): Initial model parameters with shape (n+1, 1)
    learning_rate (float): Learning rate for gradient descent
    n_epochs (int): Number of iterations over the complete dataset
    
    Returns:
    tuple: (theta, cost_history) where theta are the learned parameters
           and cost_history is a list of the cost at each epoch
    """
    m = len(y)
    cost_history = []
    
    for epoch in range(n_epochs):
        # Compute predictions
        predictions = X.dot(theta)
        
        # Compute errors
        errors = predictions - y
        
        # Compute gradient
        gradients = (1/m) * X.T.dot(errors)
        
        # Update parameters
        theta = theta - learning_rate * gradients
        
        # Compute and store cost
        cost = compute_cost(X, y, theta)
        cost_history.append(cost)
    
    return theta, cost_history

def stochastic_gradient_descent(X, y, theta, learning_rate=0.1, n_epochs=10):
    """
    Perform stochastic gradient descent.
    
    Parameters:
    X (numpy.ndarray): Input features with shape (m, n+1)
    y (numpy.ndarray): Target values with shape (m, 1)
    theta (numpy.ndarray): Initial model parameters with shape (n+1, 1)
    learning_rate (float): Learning rate for gradient descent
    n_epochs (int): Number of passes over the complete dataset
    
    Returns:
    tuple: (theta, cost_history) where theta are the learned parameters
           and cost_history is a list of the cost at each step
    """
    m = len(y)
    cost_history = []
    
    for epoch in range(n_epochs):
        # Shuffle the data
        indices = np.random.permutation(m)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        for i in range(m):
            # Get a single random training example
            xi = X_shuffled[i:i+1]
            yi = y_shuffled[i:i+1]
            
            # Compute prediction for this example
            prediction = xi.dot(theta)
            
            # Compute error for this example
            error = prediction - yi
            
            # Compute gradient for this example
            gradient = xi.T.dot(error)
            
            # Update parameters
            theta = theta - learning_rate * gradient
            
            # Compute and store cost (less frequently to save computation)
            if i % 100 == 0:
                cost = compute_cost(X, y, theta)
                cost_history.append(cost)
    
    return theta, cost_history

def mini_batch_gradient_descent(X, y, theta, learning_rate=0.1, n_epochs=10, batch_size=32):
    """
    Perform mini-batch gradient descent.
    
    Parameters:
    X (numpy.ndarray): Input features with shape (m, n+1)
    y (numpy.ndarray): Target values with shape (m, 1)
    theta (numpy.ndarray): Initial model parameters with shape (n+1, 1)
    learning_rate (float): Learning rate for gradient descent
    n_epochs (int): Number of passes over the complete dataset
    batch_size (int): Size of each mini-batch
    
    Returns:
    tuple: (theta, cost_history) where theta are the learned parameters
           and cost_history is a list of the cost at each step
    """
    m = len(y)
    cost_history = []
    n_batches = int(np.ceil(m / batch_size))
    
    for epoch in range(n_epochs):
        # Shuffle the data
        indices = np.random.permutation(m)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        for i in range(0, m, batch_size):
            # Get the current mini-batch
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            
            # Compute predictions for this mini-batch
            predictions = X_batch.dot(theta)
            
            # Compute errors
            errors = predictions - y_batch
            
            # Compute gradients
            gradients = (1/len(X_batch)) * X_batch.T.dot(errors)
            
            # Update parameters
            theta = theta - learning_rate * gradients
            
        # Compute and store cost after each epoch
        cost = compute_cost(X, y, theta)
        cost_history.append(cost)
    
    return theta, cost_history

def plot_results(X, y, theta_batch, theta_stochastic, theta_minibatch):
    """
    Plot the data points and the regression lines from different gradient descent variants.
    """
    plt.figure(figsize=(12, 8))
    
    # Plot data points
    plt.scatter(X[:, 1], y, alpha=0.5, label='Training Data')
    
    # Generate predictions for the regression lines
    X_new = np.array([[0], [2]])
    X_new_b = np.c_[np.ones((2, 1)), X_new]
    
    # Plot batch GD line
    y_predict_batch = X_new_b.dot(theta_batch)
    plt.plot(X_new, y_predict_b, 'r-', linewidth=2, label='Batch GD')
    
    # Plot stochastic GD line
    y_predict_stochastic = X_new_b.dot(theta_stochastic)
    plt.plot(X_new, y_predict_stochastic, 'g--', linewidth=2, label='Stochastic GD')
    
    # Plot mini-batch GD line
    y_predict_minibatch = X_new_b.dot(theta_minibatch)
    plt.plot(X_new, y_predict_minibatch, 'b-.', linewidth=2, label='Mini-batch GD')
    
    plt.xlabel('X', fontsize=14)
    plt.ylabel('y', fontsize=14)
    plt.title('Comparison of Gradient Descent Variants', fontsize=16)
    plt.legend(loc='upper left', fontsize=12)
    plt.axis([0, 2, 0, 15])
    plt.grid(True)
    plt.show()

def plot_cost_histories(cost_histories, labels):
    """
    Plot the cost histories for different gradient descent variants.
    """
    plt.figure(figsize=(12, 6))
    
    for cost_history, label in zip(cost_histories, labels):
        # For stochastic GD, the cost might be too noisy, so we can plot a smoothed version
        if 'Stochastic' in label:
            # Simple moving average for smoothing
            window = max(1, len(cost_history) // 50)  # Adjust window size
            weights = np.ones(window) / window
            smoothed = np.convolve(cost_history, weights, mode='valid')
            plt.plot(smoothed, label=f'{label} (smoothed)')
        else:
            plt.plot(cost_history, label=label)
    
    plt.xlabel('Iterations', fontsize=14)
    plt.ylabel('Cost', fontsize=14)
    plt.title('Cost vs. Iterations for Different Gradient Descent Variants', fontsize=16)
    plt.legend(loc='upper right', fontsize=12)
    plt.grid(True)
    plt.yscale('log')  # Use log scale for better visualization
    plt.show()

if __name__ == "__main__":
    # Reset theta for each method
    theta_batch = np.random.randn(2, 1)
    theta_stochastic = theta_batch.copy()
    theta_minibatch = theta_batch.copy()
    
    print("Initial cost:", compute_cost(X_b, y, theta_batch))
    
    # Run different gradient descent variants
    print("\nRunning Batch Gradient Descent...")
    start_time = time.time()
    theta_batch, cost_batch = batch_gradient_descent(X_b, y, theta_batch, learning_rate, n_epochs)
    print(f"Batch GD completed in {time.time() - start_time:.2f} seconds")
    print("Final parameters (Batch GD):", theta_batch.ravel())
    
    print("\nRunning Stochastic Gradient Descent...")
    start_time = time.time()
    theta_stochastic, cost_stochastic = stochastic_gradient_descent(
        X_b, y, theta_stochastic, learning_rate, n_epochs=10)
    print(f"Stochastic GD completed in {time.time() - start_time:.2f} seconds")
    print("Final parameters (Stochastic GD):", theta_stochastic.ravel())
    
    print("\nRunning Mini-batch Gradient Descent...")
    start_time = time.time()
    theta_minibatch, cost_minibatch = mini_batch_gradient_descent(
        X_b, y, theta_minibatch, learning_rate, n_epochs=10, batch_size=batch_size)
    print(f"Mini-batch GD completed in {time.time() - start_time:.2f} seconds")
    print("Final parameters (Mini-batch GD):", theta_minibatch.ravel())
    
    # Plot results
    plot_results(X_b, y, theta_batch, theta_stochastic, theta_minibatch)
    
    # Plot cost histories (with appropriate scaling for stochastic GD)
    plot_cost_histories(
        [cost_batch, cost_stochastic, cost_minibatch],
        ['Batch GD', 'Stochastic GD', 'Mini-batch GD']
    )
