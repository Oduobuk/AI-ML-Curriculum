"""
Exercise 2: Data Augmentation for CNNs

This exercise demonstrates various data augmentation techniques for CNNs
using both TensorFlow/Keras and PyTorch on the CIFAR-10 dataset.
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
torch.manual_seed(42)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create output directory for saving visualizations
import os
os.makedirs("output/augmentation_examples", exist_ok=True)

def load_cifar10():
    """Load and preprocess CIFAR-10 dataset."""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    
    # Normalize pixel values to [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Convert class vectors to one-hot encoded vectors
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    
    return (x_train, y_train), (x_test, y_test)

def create_augmentation_layers():
    """Create data augmentation layers for Keras."""
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomBrightness(0.1),
        layers.RandomContrast(0.1),
    ])

class PyTorchAugmentation:
    """Data augmentation for PyTorch."""
    def __init__(self):
        self.train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomAffine(0, translate=(0.1, 0.1)),
            transforms.RandomAffine(0, scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

def visualize_augmentations(images, num_images=5, num_augmentations=5):
    """Visualize original and augmented images."""
    # Select random images
    idx = np.random.choice(len(images), num_images, replace=False)
    selected_images = images[idx]
    
    # Create augmentation model
    data_augmentation = create_augmentation_layers()
    
    plt.figure(figsize=(15, 8))
    
    for i in range(num_images):
        # Original image
        plt.subplot(num_images, num_augmentations + 1, i * (num_augmentations + 1) + 1)
        plt.imshow(selected_images[i])
        plt.axis('off')
        if i == 0:
            plt.title('Original')
        
        # Augmented images
        for j in range(num_augmentations):
            augmented_image = data_augmentation(selected_images[i].reshape(1, 32, 32, 3))
            plt.subplot(num_images, num_augmentations + 1, i * (num_augmentations + 1) + j + 2)
            plt.imshow(augmented_image[0])
            plt.axis('off')
            if i == 0:
                plt.title(f'Aug {j+1}')
    
    plt.tight_layout()
    plt.savefig('output/augmentation_examples/data_augmentation_examples.png')
    plt.show()

def create_keras_model():
    """Create a simple CNN model for Keras."""
    return tf.keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])

class PyTorchCNN(nn.Module):
    """A simple CNN model for PyTorch."""
    def __init__(self):
        super(PyTorchCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.3),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.3)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def train_keras_model(x_train, y_train, x_val, y_val, use_augmentation=True):
    """Train a Keras model with optional data augmentation."""
    model = create_keras_model()
    
    # Compile the model
    model.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
    
    # Create data augmentation generator if enabled
    if use_augmentation:
        data_augmentation = create_augmentation_layers()
        
        # Create a data generator with augmentation
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function=data_augmentation,
            validation_split=0.2
        )
        
        # Fit the data generator
        train_generator = train_datagen.flow(
            x_train, y_train,
            batch_size=64,
            subset='training'
        )
        
        val_generator = train_datagen.flow(
            x_train, y_train,
            batch_size=64,
            subset='validation'
        )
        
        # Train the model with data augmentation
        history = model.fit(
            train_generator,
            epochs=30,
            validation_data=val_generator,
            verbose=1
        )
    else:
        # Train without data augmentation
        history = model.fit(
            x_train, y_train,
            batch_size=64,
            epochs=30,
            validation_data=(x_val, y_val),
            verbose=1
        )
    
    return model, history

def train_pytorch_model(train_loader, val_loader, use_augmentation=True):
    """Train a PyTorch model with optional data augmentation."""
    model = PyTorchCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    num_epochs = 30
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
            loss = criterion(outputs, labels.argmax(dim=1))
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.argmax(dim=1)).sum().item()
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels.argmax(dim=1))
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels.argmax(dim=1)).sum().item()
        
        val_epoch_loss = val_loss / len(val_loader.dataset)
        val_epoch_acc = val_correct / val_total
        
        # Save metrics
        history['loss'].append(epoch_loss)
        history['accuracy'].append(epoch_acc)
        history['val_loss'].append(val_epoch_loss)
        history['val_accuracy'].append(val_epoch_acc)
        
        print(f'Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc:.4f} - Val Loss: {val_epoch_loss:.4f} - Val Accuracy: {val_epoch_acc:.4f}')
    
    return model, history

def plot_comparison(history_with_aug, history_without_aug, framework):
    """Plot comparison of models with and without data augmentation."""
    plt.figure(figsize=(12, 4))
    
    # Plot training accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history_with_aug['accuracy'], label='With Augmentation')
    plt.plot(history_without_aug['accuracy'], label='Without Augmentation')
    plt.title(f'{framework} Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history_with_aug['val_accuracy'], label='With Augmentation')
    plt.plot(history_without_aug['val_accuracy'], label='Without Augmentation')
    plt.title(f'{framework} Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'output/augmentation_comparison_{framework.lower()}.png')
    plt.show()

def main():
    # Load and preprocess data
    print("Loading CIFAR-10 dataset...")
    (x_train, y_train), (x_test, y_test) = load_cifar10()
    
    # Split training data into training and validation sets
    from sklearn.model_selection import train_test_split
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
    
    # Visualize data augmentation
    print("\nVisualizing data augmentation...")
    visualize_augmentations(x_train)
    
    # Keras Experiment
    print("\nRunning Keras experiment...")
    print("Training with data augmentation...")
    keras_model_with_aug, keras_history_with_aug = train_keras_model(x_train, y_train, x_val, y_val, use_augmentation=True)
    
    print("\nTraining without data augmentation...")
    keras_model_without_aug, keras_history_without_aug = train_keras_model(x_train, y_train, x_val, y_val, use_augmentation=False)
    
    # Plot comparison for Keras
    print("\nGenerating Keras comparison plot...")
    plot_comparison(keras_history_with_aug.history, keras_history_without_aug.history, 'Keras')
    
    # PyTorch Experiment
    print("\nRunning PyTorch experiment...")
    # Create PyTorch datasets
    class CIFAR10Dataset(torch.utils.data.Dataset):
        def __init__(self, x, y, transform=None):
            self.x = x
            self.y = y
            self.transform = transform
        
        def __len__(self):
            return len(self.x)
        
        def __getitem__(self, idx):
            x = self.x[idx]
            y = self.y[idx]
            
            if self.transform:
                x = self.transform(x)
            else:
                x = torch.tensor(x.transpose(2, 0, 1), dtype=torch.float32)
            
            return x, torch.tensor(y, dtype=torch.float32)
    
    # Create data loaders with and without augmentation
    augmentation = PyTorchAugmentation()
    
    # With augmentation
    train_dataset_aug = CIFAR10Dataset(x_train, y_train, transform=augmentation.train_transform)
    val_dataset = CIFAR10Dataset(x_val, y_val, transform=augmentation.test_transform)
    
    # Without augmentation
    train_dataset_no_aug = CIFAR10Dataset(x_train, y_train, transform=augmentation.test_transform)
    
    # Create data loaders
    batch_size = 64
    train_loader_aug = DataLoader(train_dataset_aug, batch_size=batch_size, shuffle=True)
    train_loader_no_aug = DataLoader(train_dataset_no_aug, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Train with augmentation
    print("Training with data augmentation...")
    pytorch_model_with_aug, pytorch_history_with_aug = train_pytorch_model(train_loader_aug, val_loader, use_augmentation=True)
    
    # Train without augmentation
    print("\nTraining without data augmentation...")
    pytorch_model_without_aug, pytorch_history_without_aug = train_pytorch_model(train_loader_no_aug, val_loader, use_augmentation=False)
    
    # Plot comparison for PyTorch
    print("\nGenerating PyTorch comparison plot...")
    plot_comparison(pytorch_history_with_aug, pytorch_history_without_aug, 'PyTorch')
    
    print("\nData augmentation experiment completed!")
    print("Check the 'output' directory for visualizations and results.")

if __name__ == "__main__":
    main()
