Famous Computer Vision Architectures

This document provides a conceptual overview of some of the most influential and widely used convolutional neural network (CNN) architectures in computer vision.

 Inception

The Inception architecture, developed by Google, was designed to achieve high accuracy while being computationally efficient. The key innovation of the Inception architecture is the "Inception module," which performs multiple different convolutions (e.g., 1x1, 3x3, 5x5) and pooling operations in parallel on the same input and concatenates the results. This allows the network to capture features at different scales.

 Key Features of Inception:

    Multi-scale feature extraction: By using filters of different sizes, the Inception module can capture features at various scales simultaneously.
    Dimensionality reduction: The Inception module uses 1x1 convolutions to reduce the number of channels before applying more expensive 3x3 and 5x5 convolutions, which helps to reduce the computational cost.
    Improved performance: The Inception architecture has achieved state-of-the-art performance on many computer vision benchmarks, including the ImageNet Large Scale Visual Recognition Challenge (ILSVRC).

 MobileNet

The MobileNet family of models is specifically designed for mobile and embedded vision applications where computational resources are limited. The core innovation of MobileNet is the use of depthwise separable convolutions.

 Key Features of MobileNet:

    Depthwise separable convolutions: This technique splits a standard convolution into two parts: a depthwise convolution (which acts on each input channel separately) and a pointwise convolution (a 1x1 convolution to combine the outputs). This factorization significantly reduces the number of parameters and computations, resulting in a lightweight and fast model.
    Width and resolution multipliers: MobileNet introduces two hyperparameters, the width multiplier and the resolution multiplier, which allow you to trade off accuracy for latency and model size. The width multiplier thins the network, while the resolution multiplier reduces the input image resolution.
    Efficiency: MobileNets are highly efficient and can be run in real-time on mobile devices with limited computational resources.