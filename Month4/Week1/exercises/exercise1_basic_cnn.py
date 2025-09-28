"""
Exercise 1: Basic CNN Implementation

This exercise demonstrates how to implement a simple CNN for image classification
using both TensorFlow/Keras and PyTorch on the CIFAR-10 dataset.
"""

# Standard library imports
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Set random seeds for reproducibility
np.random.seed(42)
import tensorflow as tf
tf.random.set_seed(42)
import torch
import torch.nn as nn
import torch.optim as optim
torch.manual_seed(42)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create output directory for saving models and plots
os.makedirs("output", exist_ok=True)

def load_cifar10():
    """Load and preprocess CIFAR-10 dataset."""
    # Load CIFAR-10 data
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    
    # Normalize pixel values to [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Convert class vectors to one-hot encoded vectors
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    
    return (x_train, y_train), (x_test, y_test)

def plot_training_history(history, framework):
    """Plot training and validation accuracy/loss."""
    plt.figure(figsize=(12, 4))
    
    # Plot training & validation accuracy values
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'] if framework == 'keras' else history['accuracy'])
    plt.plot(history.history['val_accuracy'] if framework == 'keras' else history['val_accuracy'])
    plt.title(f'{framework.upper()} Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'] if framework == 'keras' else history['loss'])
    plt.plot(history.history['val_loss'] if framework == 'keras' else history['val_loss'])
    plt.title(f'{framework.upper()} Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig(f'output/{framework}_training_history.png')
    plt.show()

def create_keras_model():
    """Create a simple CNN model using Keras."""
    model = tf.keras.Sequential([
        # First Convolutional Block
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.2),
        
        # Second Convolutional Block
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.3),
        
        # Third Convolutional Block
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.4),
        
        # Dense Layers
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    return model

class PyTorchCNN(nn.Module):
    """A simple CNN model using PyTorch."""
    def __init__(self):
        super(PyTorchCNN, self).__init__()
        # First Convolutional Block
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.2)
        )
        
        # Second Convolutional Block
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.3)
        )
        
        # Third Convolutional Block
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.4)
        )
        
        # Dense Layers
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.classifier(x)
        return x

def train_pytorch_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=25):
    """Train the PyTorch model."""
    history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == torch.argmax(labels, dim=1)).sum().item()
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        
        # Validation phase
        val_loss, val_acc = evaluate_pytorch_model(model, val_loader, criterion)
        
        # Save metrics
        history['loss'].append(epoch_loss)
        history['accuracy'].append(epoch_acc)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_acc)
        
        print(f'Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc:.4f} - Val Loss: {val_loss:.4f} - Val Accuracy: {val_acc:.4f}')
    
    return history

def evaluate_pytorch_model(model, test_loader, criterion):
    """Evaluate the PyTorch model on test data."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == torch.argmax(labels, dim=1)).sum().item()
    
    loss = running_loss / len(test_loader.dataset)
    accuracy = correct / total
    
    return loss, accuracy

def main():
    # Load and preprocess data
    print("Loading CIFAR-10 dataset...")
    (x_train, y_train), (x_test, y_test) = load_cifar10()
    
    # Split training data into training and validation sets
    from sklearn.model_selection import train_test_split
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
    
    # Create data loaders for PyTorch
    batch_size = 64
    
    # Convert numpy arrays to PyTorch tensors
    x_train_torch = torch.tensor(x_train.transpose(0, 3, 1, 2), dtype=torch.float32)
    y_train_torch = torch.tensor(y_train, dtype=torch.long)
    x_val_torch = torch.tensor(x_val.transpose(0, 3, 1, 2), dtype=torch.float32)
    y_val_torch = torch.tensor(y_val, dtype=torch.long)
    
    # Create PyTorch datasets and data loaders
    from torch.utils.data import TensorDataset, DataLoader
    
    train_dataset = TensorDataset(x_train_torch, y_train_torch)
    val_dataset = TensorDataset(x_val_torch, y_val_torch)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Train Keras model
    print("\nTraining Keras model...")
    keras_model = create_keras_model()
    
    # Compile the model
    keras_model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
    
    # Train the model
    keras_history = keras_model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=25,
        validation_data=(x_val, y_val),
        verbose=1
    )
    
    # Evaluate Keras model
    print("\nEvaluating Keras model...")
    test_loss, test_acc = keras_model.evaluate(x_test, y_test, verbose=0)
    print(f'Keras Model - Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}')
    
    # Plot training history for Keras model
    plot_training_history(keras_history, 'keras')
    
    # Save Keras model
    keras_model.save('output/keras_cifar10_model.h5')
    
    # Train PyTorch model
    print("\nTraining PyTorch model...")
    pytorch_model = PyTorchCNN().to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(pytorch_model.parameters(), lr=0.001)
    
    # Train the model
    pytorch_history = train_pytorch_model(
        pytorch_model, train_loader, val_loader, criterion, optimizer, num_epochs=25
    )
    
    # Evaluate PyTorch model
    print("\nEvaluating PyTorch model...")
    test_dataset = TensorDataset(
        torch.tensor(x_test.transpose(0, 3, 1, 2), dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long)
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    test_loss, test_acc = evaluate_pytorch_model(pytorch_model, test_loader, criterion)
    print(f'PyTorch Model - Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}')
    
    # Plot training history for PyTorch model
    plot_training_history(pytorch_history, 'pytorch')
    
    # Save PyTorch model
    torch.save(pytorch_model.state_dict(), 'output/pytorch_cifar10_model.pth')

if __name__ == "__main__":
    main()
