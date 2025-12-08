# Datasets for Generative Models

Generative models, especially those focused on image generation, often require large and diverse image datasets. The choice of dataset depends on the complexity of the model you are building and the type of images you want to generate.

## Recommended Datasets

### For Beginners & Simpler Models (AEs, VAEs)

1.  **MNIST (Modified National Institute of Standards and Technology database)**
    *   **Description:** A dataset of 70,000 grayscale images of handwritten digits (0-9). Each image is 28x28 pixels.
    *   **Why it's useful:** Small, simple, and widely used for introductory deep learning tasks. Excellent for quickly testing Autoencoders and VAEs.
    *   **Access:** Available directly in deep learning libraries like TensorFlow/Keras and PyTorch.

2.  **Fashion MNIST**
    *   **Description:** A dataset of 70,000 grayscale images of fashion products (e.g., shirts, trousers, sneakers). Each image is 28x28 pixels. It's a direct drop-in replacement for MNIST, but more challenging.
    *   **Why it's useful:** Provides a slightly more complex task than MNIST while maintaining the same image dimensions, making it suitable for benchmarking.
    *   **Access:** Available directly in deep learning libraries like TensorFlow/Keras and PyTorch.

### For More Advanced Models (GANs, VAEs)

1.  **CIFAR-10 / CIFAR-100**
    *   **Description:** CIFAR-10 consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. CIFAR-100 is similar but has 100 classes.
    *   **Why it's useful:** Introduces color images and more diverse object categories, making it a good step up for generative models.
    *   **Access:** Available directly in deep learning libraries.

2.  **CelebA (Large-scale CelebFaces Attributes Dataset)**
    *   **Description:** A large-scale face attributes dataset with over 200K celebrity images, each with 40 attribute annotations.
    *   **Why it's useful:** Ideal for training GANs to generate realistic human faces, and for tasks like attribute manipulation or style transfer.
    *   **Access:** Can be downloaded from the official project page or often found pre-processed in various ML repositories.

3.  **LSUN (Large-scale Scene Understanding) Dataset**
    *   **Description:** A large-scale image dataset with millions of labeled images of various scenes and objects.
    *   **Why it's useful:** Provides a very diverse and high-resolution set of images for training state-of-the-art generative models.
    *   **Access:** Requires downloading from the official LSUN project page.

## Considerations for Generative Models

*   **Image Resolution:** Higher resolution images require more computational resources and more complex models.
*   **Dataset Size:** Generative models often benefit from larger datasets to learn diverse patterns.
*   **Data Preprocessing:** Normalizing pixel values (e.g., to [-1, 1] for GANs or [0, 1] for AEs/VAEs) is crucial.
*   **Computational Resources:** Training advanced generative models like GANs can be very computationally intensive.
