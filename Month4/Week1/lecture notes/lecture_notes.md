# Lecture Notes: Convolutional Neural Networks (CNNs) for Computer Vision

## Table of Contents
1. [Introduction to CNNs](#introduction-to-cnns)
2. [Convolution Operation](#convolution-operation)
3. [CNN Architectures](#cnn-architectures)
4. [Training CNNs](#training-cnns)
5. [Transfer Learning](#transfer-learning)
6. [Practical Considerations](#practical-considerations)
7. [Advanced Topics](#advanced-topics)
8. [Resources](#resources)

## Introduction to CNNs

### What are CNNs?
- Specialized neural networks for processing grid-like data (images, time-series)
- Inspired by the visual cortex in animals
- Key properties:
  - Local connectivity
  - Shared weights
  - Pooling operations
  - Hierarchical feature learning

### Why CNNs for Computer Vision?
- Traditional neural networks don't scale well to high-dimensional image data
- CNNs exploit spatial locality and translation invariance
- Significantly fewer parameters than fully-connected networks
- Better generalization on visual data

## Convolution Operation

### Basic Concepts
- **Convolution**: Mathematical operation that combines two functions to produce a third function
- **Kernel/Filter**: Small matrix used to extract features from the input
- **Stride**: Step size of the kernel when moving across the image
- **Padding**: Adding zeros around the input to control spatial dimensions

### Types of Convolutions
1. **Standard Convolution**
   - Most common type
   - Kernel slides across the input with a fixed stride

2. **Dilated/Atrous Convolution**
   - Expands the receptive field without increasing parameters
   - Uses gaps between kernel elements

3. **Transposed Convolution**
   - Also known as deconvolution or fractionally-strided convolution
   - Used for upsampling in segmentation and generation tasks

4. **Depthwise Separable Convolution**
   - More efficient than standard convolution
   - Separates spatial and channel-wise operations

### Pooling Layers
- **Max Pooling**: Takes the maximum value in each window
- **Average Pooling**: Takes the average value in each window
- **Global Average Pooling**: Takes the average of each feature map
- **Purpose**:
  - Dimensionality reduction
  - Translation invariance
  - Reducing computational complexity

## CNN Architectures

### LeNet-5 (1998)
- One of the first successful CNNs
- Architecture:
  - 2 Convolutional layers
  - 2 Subsampling (pooling) layers
  - 3 Fully-connected layers
- Used for digit recognition (MNIST)

### AlexNet (2012)
- Breakthrough in ImageNet competition
- Key features:
  - ReLU activation
  - Dropout
  - Data augmentation
  - Local response normalization
  - GPU implementation

### VGG (2014)
- Simple and uniform architecture
- Key features:
  - Small 3x3 filters throughout
  - Increasing number of filters
  - 2-3 convolutional layers between pooling

### ResNet (2015)
- Introduced residual connections (skip connections)
- Solves the vanishing gradient problem in deep networks
- Key features:
  - Identity mapping with skip connections
  - Batch normalization
  - Bottleneck blocks for deeper networks

### EfficientNet (2019)
- Compound scaling method
- Balances network depth, width, and resolution
- More efficient than previous architectures

## Training CNNs

### Data Augmentation
- **Geometric Transformations**:
  - Random rotations
  - Flips
  - Translations
  - Zooming
- **Color Space Transformations**:
  - Brightness adjustment
  - Contrast adjustment
  - Saturation adjustment
  - Hue adjustment
- **Advanced Techniques**:
  - Cutout
  - Mixup
  - CutMix
  - AutoAugment

### Optimization
- **Optimizers**:
  - SGD with momentum
  - Adam
  - RMSprop
- **Learning Rate Scheduling**:
  - Step decay
  - Cosine annealing
  - One-cycle learning rate
- **Regularization**:
  - L1/L2 regularization
  - Dropout
  - Batch normalization
  - Weight decay

### Batch Normalization
- Normalizes the activations of the previous layer
- Benefits:
  - Faster training
  - Higher learning rates
  - Less sensitive to initialization
  - Acts as regularization

## Transfer Learning

### Why Transfer Learning?
- Limited labeled data
- Computational efficiency
- Better performance with less data

### Approaches
1. **Feature Extraction**:
   - Use pre-trained CNN as a fixed feature extractor
   - Replace the final classification layer
   - Train only the new layers

2. **Fine-tuning**:
   - Unfreeze some of the top layers
   - Train both the new layers and the unfrozen layers
   - Use a lower learning rate

### Popular Pre-trained Models
- VGG16/VGG19
- ResNet50/ResNet101
- InceptionV3
- EfficientNet
- Vision Transformer (ViT)

## Practical Considerations

### Input Pipeline
- Use `tf.data` or PyTorch `DataLoader`
- Prefetching and caching
- Parallel data loading
- Data augmentation on the GPU

### Mixed Precision Training
- Use 16-bit floating-point numbers
- Faster training
- Less memory usage
- Minimal impact on accuracy

### Model Deployment
- Model optimization (quantization, pruning)
- Conversion to optimized formats (TensorRT, ONNX, TFLite)
- Serving with TensorFlow Serving or TorchServe

## Advanced Topics

### Object Detection
- R-CNN family (R-CNN, Fast R-CNN, Faster R-CNN)
- YOLO (You Only Look Once)
- SSD (Single Shot MultiBox Detector)
- EfficientDet

### Semantic Segmentation
- FCN (Fully Convolutional Networks)
- U-Net
- DeepLab
- Mask R-CNN

### Generative Models
- GANs (Generative Adversarial Networks)
- VAEs (Variational Autoencoders)
- Diffusion Models

## Resources

### Books
- "Deep Learning" by Ian Goodfellow, Yoshua Bengio, and Aaron Courville
- "Deep Learning for Computer Vision" by Rajalingappaa Shanmugamani
- "Computer Vision: Algorithms and Applications" by Richard Szeliski

### Online Courses
- [CS231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/)
- [Fast.ai Practical Deep Learning for Coders](https://course.fast.ai/)
- [DeepLearning.AI TensorFlow Developer Specialization](https://www.coursera.org/professional-certificates/tensorflow-in-practice)

### Research Papers
- [ImageNet Classification with Deep Convolutional Neural Networks (AlexNet)](https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)
- [Very Deep Convolutional Networks for Large-Scale Image Recognition (VGG)](https://arxiv.org/abs/1409.1556)
- [Deep Residual Learning for Image Recognition (ResNet)](https://arxiv.org/abs/1512.03385)
- [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)

### Libraries and Frameworks
- [TensorFlow](https://www.tensorflow.org/)
- [PyTorch](https://pytorch.org/)
- [Keras](https://keras.io/)
- [Fast.ai](https://www.fast.ai/)
- [Albumentations](https://albumentations.ai/) (for image augmentations)
