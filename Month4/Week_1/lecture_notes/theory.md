Week 1: Convolutional Neural Networks (CNNs) for Computer Vision

 Introduction to Convolutional Neural Networks (CNNs)

A Convolutional Neural Network (CNN or ConvNet) is a deep learning algorithm particularly suited for tasks involving object recognition, such as image classification, detection, and segmentation. CNNs are designed to automatically and adaptively learn spatial hierarchies of features from input data.

 Key Components of a CNN

A typical CNN architecture is composed of four main types of layers:

1.  Convolutional Layers: These are the core building blocks of a CNN. They apply a set of learnable filters (or kernels) to the input image to extract features. As the filter slides over the input, it performs a dot product to create a feature map. This process allows the network to detect patterns like edges, corners, and textures.

2.  Activation Function (ReLU): After a convolutional layer, an activation function is applied element-wise to the feature map. The most common activation function is the Rectified Linear Unit (ReLU), which introduces non-linearity into the model and helps it learn more complex patterns.

3.  Pooling Layers: These layers are used to downsample the feature maps, reducing their spatial dimensions. This reduces the computational load, memory usage, and also helps to prevent overfitting. The most common types of pooling are max pooling and average pooling.

4.  Fully Connected Layers: These are typically the final layers in a CNN. They take the high-level features extracted by the convolutional and pooling layers and use them to perform the classification task.