"""
Exercise 3: CNN Visualization and Interpretation

This exercise demonstrates how to visualize and interpret what CNNs learn
using techniques like filter visualization, activation maximization, and Grad-CAM.
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers, applications
from tensorflow.keras.preprocessing import image
import cv2
import os

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Create output directory for saving visualizations
os.makedirs("output/cnn_visualization", exist_ok=True)

def load_pretrained_model():
    """Load a pre-trained VGG16 model for visualization."""
    # Load VGG16 pre-trained on ImageNet
    base_model = applications.VGG16(weights='imagenet', include_top=True)
    
    # Create a model that outputs the activations of each layer
    layer_outputs = [layer.output for layer in base_model.layers[:8]]  # First 8 layers
    activation_model = models.Model(inputs=base_model.input, outputs=layer_outputs)
    
    return base_model, activation_model

def load_and_preprocess_image(img_path, target_size=(224, 224)):
    """Load and preprocess an image for the VGG16 model."""
    # Load image and resize to target size
    img = image.load_img(img_path, target_size=target_size)
    
    # Convert to array and preprocess for VGG16
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = applications.vgg16.preprocess_input(img_array)
    
    return img, img_array

def visualize_filters(model, layer_name, num_filters=16):
    """Visualize the filters of a specific layer."""
    # Get the layer
    layer = model.get_layer(name=layer_name)
    
    # Get the filters (weights) of the layer
    filters, biases = layer.get_weights()
    
    # Normalize filter values to 0-1 for visualization
    f_min, f_max = filters.min(), filters.max()
    filters = (filters - f_min) / (f_max - f_min)
    
    # Plot the filters
    n_filters = min(num_filters, filters.shape[3])
    fig = plt.figure(figsize=(12, 8))
    
    for i in range(n_filters):
        # Get the filter
        f = filters[:, :, :, i]
        
        # Plot each channel separately
        for j in range(filters.shape[2]):
            ax = plt.subplot(n_filters, filters.shape[2], i * filters.shape[2] + j + 1)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.imshow(f[:, :, j], cmap='viridis')
    
    plt.suptitle(f'Filters in layer: {layer_name}')
    plt.tight_layout()
    plt.savefig(f'output/cnn_visualization/filters_{layer_name}.png')
    plt.show()

def visualize_activations(img_array, model, layer_names):
    """Visualize the activations of specified layers for a given input image."""
    # Get the outputs of the specified layers
    layer_outputs = [model.get_layer(name).output for name in layer_names]
    
    # Create a model that will return these outputs, given the model input
    activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
    
    # Get the activations
    activations = activation_model.predict(img_array)
    
    # Visualize each activation channel for each layer
    for layer_name, layer_activation in zip(layer_names, activations):
        # Number of features in the feature map
        n_features = layer_activation.shape[-1]
        
        # The feature map has shape (1, size, size, n_features)
        size = layer_activation.shape[1]
        
        # We will tile the activation channels in this matrix
        n_cols = 8
        n_rows = n_features // n_cols
        
        # Initialize the output image
        display_grid = np.zeros((size * n_rows, size * n_cols))
        
        # Fill the grid with the activations
        for row in range(n_rows):
            for col in range(n_cols):
                channel_image = layer_activation[0, :, :, row * n_cols + col]
                
                # Post-process the feature to make it visually palatable
                channel_image -= channel_image.mean()
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                
                # Insert into the grid
                display_grid[row * size: (row + 1) * size,
                            col * size: (col + 1) * size] = channel_image
        
        # Display the grid
        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1],
                           scale * display_grid.shape[0]))
        plt.title(f'Activations for {layer_name}')
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')
        plt.savefig(f'output/cnn_visualization/activations_{layer_name}.png')
        plt.show()

def generate_gradcam(model, img_array, layer_name, pred_index=None):
    """Generate a Grad-CAM heatmap for a specific class prediction."""
    # Create a model that maps the input image to the activations of the last conv layer
    grad_model = models.Model(
        inputs=model.input,
        outputs=[model.get_layer(layer_name).output, model.output]
    )
    
    # Compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        loss = predictions[:, pred_index]
    
    # Extract filters and gradients
    output = conv_outputs[0]
    grads = tape.gradient(loss, conv_outputs)[0]
    
    # Global average pooling of the gradients
    weights = tf.reduce_mean(grads, axis=(0, 1))
    
    # Build a weighted map of the filters according to the gradients
    cam = np.zeros(output.shape[0:2], dtype=np.float32)
    
    for i, w in enumerate(weights):
        cam += w * output[:, :, i]
    
    # Apply ReLU to the class activation map
    cam = np.maximum(cam, 0)
    
    # Normalize the heatmap
    cam = cam / np.max(cam)
    
    # Resize the heatmap to the original image size
    cam = cv2.resize(cam, (img_array.shape[2], img_array.shape[1]))
    
    # Convert to 0-255 range
    cam = np.uint8(255 * cam)
    
    # Create heatmap
    heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    
    # Convert back to RGB
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    return heatmap, cam

def visualize_gradcam(img, img_array, model, layer_name, pred_index=None):
    """Visualize Grad-CAM heatmap on top of the original image."""
    # Generate Grad-CAM heatmap
    heatmap, _ = generate_gradcam(model, img_array, layer_name, pred_index)
    
    # Resize heatmap to match the original image size
    heatmap = cv2.resize(heatmap, (img_array.shape[2], img_array.shape[1]))
    
    # Convert the image to RGB if it's grayscale
    if len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    
    # Superimpose the heatmap on the original image
    superimposed_img = heatmap * 0.4 + img_array[0] * 255
    superimposed_img = np.clip(superimposed_img, 0, 255).astype('uint8')
    
    # Display the original image and the heatmap
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(heatmap)
    plt.title('Grad-CAM Heatmap')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(superimposed_img / 255.0)
    plt.title('Superimposed')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'output/cnn_visualization/gradcam_{layer_name}.png')
    plt.show()

def visualize_feature_maps(model, img_array, layer_name):
    """Visualize feature maps for a given input image and layer."""
    # Create a model that will return the outputs of the specified layer
    layer_output = model.get_layer(layer_name).output
    activation_model = models.Model(inputs=model.input, outputs=layer_output)
    
    # Get the feature maps
    feature_maps = activation_model.predict(img_array)
    
    # Number of features in the feature map
    n_features = feature_maps.shape[-1]
    
    # The feature map has shape (1, size, size, n_features)
    size = feature_maps.shape[1]
    
    # We will tile the activation channels in this matrix
    n_cols = 8
    n_rows = n_features // n_cols + (1 if n_features % n_cols != 0 else 0)
    
    # Initialize the output image
    display_grid = np.zeros((size * n_rows, size * n_cols))
    
    # Fill the grid with the feature maps
    for row in range(n_rows):
        for col in range(n_cols):
            channel_index = row * n_cols + col
            
            # Stop if we've gone through all the feature maps
            if channel_index >= n_features:
                break
                
            # Get the feature map
            feature_map = feature_maps[0, :, :, channel_index]
            
            # Normalize the feature map
            feature_map -= feature_map.mean()
            if feature_map.std() > 0:
                feature_map /= feature_map.std()
            feature_map *= 64
            feature_map += 128
            feature_map = np.clip(feature_map, 0, 255).astype('uint8')
            
            # Insert into the grid
            display_grid[row * size: (row + 1) * size,
                        col * size: (col + 1) * size] = feature_map
    
    # Display the grid
    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1],
                       scale * display_grid.shape[0]))
    plt.title(f'Feature maps for {layer_name}')
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')
    plt.savefig(f'output/cnn_visualization/feature_maps_{layer_name}.png')
    plt.show()

def main():
    # Load a pre-trained model
    print("Loading pre-trained VGG16 model...")
    model, _ = load_pretrained_model()
    
    # Load and preprocess an example image
    print("Loading example image...")
    img_path = tf.keras.utils.get_file(
        'elephant.jpg',
        'https://storage.googleapis.com/tensorflow/tf-keras-datasets/elephant.jpg'
    )
    
    img, img_array = load_and_preprocess_image(img_path)
    
    # Visualize filters in the first convolutional layer
    print("Visualizing filters...")
    visualize_filters(model, 'block1_conv1', num_filters=16)
    
    # Visualize activations for different layers
    print("Visualizing activations...")
    layer_names = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1']
    visualize_activations(img_array, model, layer_names)
    
    # Visualize feature maps
    print("Visualizing feature maps...")
    for layer_name in layer_names:
        visualize_feature_maps(model, img_array, layer_name)
    
    # Generate and visualize Grad-CAM
    print("Generating Grad-CAM visualizations...")
    layer_name = 'block5_conv3'  # Last convolutional layer in VGG16
    visualize_gradcam(img, img_array, model, layer_name)
    
    print("\nVisualization completed!")
    print("Check the 'output/cnn_visualization' directory for the results.")

if __name__ == "__main__":
    main()
