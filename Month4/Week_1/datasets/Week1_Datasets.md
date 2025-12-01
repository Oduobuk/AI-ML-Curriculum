# Month 4, Week 1: Foundational Datasets for Computer Vision

When learning about Convolutional Neural Networks, it's essential to be familiar with the standard datasets that have driven much of the research and progress in the field. These datasets serve as benchmarks for testing new architectures and algorithms.

## 1. MNIST

*   **Description:** The "Hello, World!" of computer vision. It's a dataset of 70,000 grayscale images of handwritten digits (0-9). The images are small and clean, making it perfect for testing a new model implementation.
*   **Image Size:** 28x28x1 (grayscale)
*   **Classes:** 10 (the digits 0 through 9)
*   **Use Case:** Excellent for a first-time implementation of a simple CNN. You can achieve very high accuracy (>99%) with a basic architecture.

## 2. CIFAR-10

*   **Description:** A more challenging dataset than MNIST, and a very common benchmark for new ideas. It consists of 60,000 small color images across 10 different classes. The images are low-resolution, which makes the task non-trivial.
*   **Image Size:** 32x32x3 (RGB color)
*   **Classes:** 10 (airplane, car, bird, cat, deer, dog, frog, horse, ship, truck)
*   **Use Case:** A great "next step" after MNIST. It's complex enough to require a real CNN architecture but small enough to be trained relatively quickly on a modern GPU.

## 3. CIFAR-100

*   **Description:** A direct extension of CIFAR-10. It has the same number and size of images but contains 100 different classes. This makes the classification task significantly harder. The 100 classes are also grouped into 20 "superclasses."
*   **Image Size:** 32x32x3 (RGB color)
*   **Classes:** 100
*   **Use Case:** Used for testing a model's ability to handle a larger number of more fine-grained classes.

## 4. ImageNet

*   **Description:** The dataset that arguably launched the modern deep learning revolution. The ImageNet Large Scale Visual Recognition Challenge (ILSVRC) is an annual competition that uses a subset of the full ImageNet dataset. It contains over 1.2 million high-resolution training images.
*   **Image Size:** Variable, but typically resized to 224x224x3 for training.
*   **Classes:** 1,000 (e.g., specific breeds of dogs, types of cars, many different animals and objects).
*   **Use Case:** The standard benchmark for large-scale image classification. Training a model from scratch on ImageNet is a major computational undertaking and is typically done by research labs and large companies. More commonly, developers will use models that have been **pre-trained** on ImageNet and adapt them for their own tasks (a technique called Transfer Learning, which will be covered in Week 2).

These datasets are all publicly available and are integrated into deep learning frameworks like TensorFlow and PyTorch, making them easy to download and use.
