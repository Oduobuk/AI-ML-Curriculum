# Month 4: Deep Learning for Vision and Language

## Week 1: Convolutional Neural Networks (CNNs) for Computer Vision

### Learning Objectives
By the end of this week, you will be able to:
- Understand the fundamental concepts of Convolutional Neural Networks (CNNs)
- Implement CNNs using both TensorFlow/Keras and PyTorch
- Train and evaluate CNNs on image classification tasks
- Visualize and interpret what CNNs learn
- Apply data augmentation techniques to improve model generalization

### Prerequisites
- Basic understanding of neural networks (from Month 3, Week 4)
- Familiarity with Python and NumPy
- Basic knowledge of linear algebra and calculus

### Weekly Schedule

#### Day 1: Introduction to CNNs
- **Morning**: Theory of CNNs
  - Biological inspiration: Visual cortex
  - Convolution operation
  - Pooling layers
  - Strides and padding
- **Afternoon**: Implementing Basic CNNs
  - Building blocks in Keras/PyTorch
  - MNIST classification example

#### Day 2: CNN Architectures
- **Morning**: Common CNN Architectures
  - LeNet-5
  - AlexNet
  - VGGNet
- **Afternoon**: Hands-on with CNN Architectures
  - Implementing VGG-like networks
  - Model visualization

#### Day 3: Training CNNs
- **Morning**: Training Techniques
  - Data augmentation
  - Batch normalization
  - Dropout
  - Learning rate scheduling
- **Afternoon**: Practical Session
  - Implementing data augmentation
  - Training on CIFAR-10

#### Day 4: CNN Visualization and Interpretation
- **Morning**: Understanding What CNNs Learn
  - Feature visualization
  - Filter visualization
  - Class activation maps (Grad-CAM)
- **Afternoon**: Practical Session
  - Visualizing filters and activations
  - Implementing Grad-CAM

#### Day 5: Project Work
- **Full Day**: Image Classification Project
  - Work on a custom image classification task
  - Apply techniques learned during the week
  - Present results and findings

### Exercises
1. **Basic CNN Implementation**
   - Build a CNN from scratch using both Keras and PyTorch
   - Train on the CIFAR-10 dataset
   - Compare performance with a simple fully-connected network

2. **Data Augmentation**
   - Implement various data augmentation techniques
   - Measure the impact on model performance
   - Visualize augmented images

3. **CNN Visualization**
   - Visualize filters and feature maps
   - Implement Grad-CAM to understand model decisions

### Project
**CIFAR-10 Classification Challenge**
- Build a CNN model to classify images from the CIFAR-10 dataset
- Implement data augmentation
- Achieve at least 85% test accuracy
- Visualize model predictions and mistakes

### Resources
- [CS231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/)
- [Deep Learning Book - Chapter 9: Convolutional Networks](https://www.deeplearningbook.org/contents/convnets.html)
- [A Comprehensive Guide to Convolutional Neural Networks](https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53)

### Assessment
- Weekly quiz on CNN concepts (20%)
- Code exercises (40%)
- End-of-week project (40%)
