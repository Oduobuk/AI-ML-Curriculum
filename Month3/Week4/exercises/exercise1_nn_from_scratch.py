"""
Exercise 1: Building a Neural Network from Scratch
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2
    
    def train(self, X, y, learning_rate=0.1, epochs=10000):
        for _ in range(epochs):
            # Forward pass
            output = self.forward(X)
            
            # Backward pass
            m = X.shape[0]
            dz2 = output - y.reshape(-1, 1)
            dW2 = np.dot(self.a1.T, dz2) / m
            db2 = np.sum(dz2, axis=0, keepdims=True) / m
            
            dz1 = np.dot(dz2, self.W2.T) * (self.a1 * (1 - self.a1))
            dW1 = np.dot(X.T, dz1) / m
            db1 = np.sum(dz1, axis=0, keepdims=True) / m
            
            # Update parameters
            self.W2 -= learning_rate * dW2
            self.b2 -= learning_rate * db2
            self.W1 -= learning_rate * dW1
            self.b1 -= learning_rate * db1
    
    def predict(self, X, threshold=0.5):
        return (self.forward(X) > threshold).astype(int)

def plot_decision_boundary(model, X, y):
    # Create mesh grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    
    # Predict for each point in the grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Decision Boundary')
    plt.show()

def main():
    # Generate moon dataset
    X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
    
    # Create and train the neural network
    nn = NeuralNetwork(input_size=2, hidden_size=4, output_size=1)
    nn.train(X, y, learning_rate=0.1, epochs=10000)
    
    # Plot decision boundary
    plot_decision_boundary(nn, X, y)
    
    # Calculate accuracy
    predictions = nn.predict(X)
    accuracy = np.mean(predictions == y.reshape(-1, 1))
    print(f"Training accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()
