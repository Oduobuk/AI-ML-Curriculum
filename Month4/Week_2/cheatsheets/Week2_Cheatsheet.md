# Month 4, Week 2 Cheatsheet: Advanced CV & Transfer Learning

## Transfer Learning Fundamentals

*   **Definition:** Reusing a pre-trained model (often a large CNN trained on ImageNet) as a starting point for a new, related task.
*   **Why use it?**
    *   Reduces need for massive datasets.
    *   Saves computational resources and time.
    *   Leverages generic, low-level features learned by the pre-trained model.
*   **Pre-trained Models:** CNNs trained on large datasets like ImageNet (e.g., VGG, ResNet, Inception, MobileNet). These models have learned a rich hierarchy of features.

---

## Transfer Learning Strategies

### 1. Feature Extraction (Fixed Feature Extractor)

*   **Process:**
    1.  Load a pre-trained model.
    2.  Remove its original classification head (final dense layers).
    3.  **Freeze** the weights of the convolutional base (feature extractor).
    4.  Add a new, custom classification head (e.g., `Dense` layers) on top.
    5.  Train **only the new classification head** on your dataset.
*   **When to use:**
    *   Your dataset is **small**.
    *   Your dataset is **similar** to the original dataset the model was trained on.
*   **Analogy:** Using a powerful microscope (pre-trained feature extractor) to analyze new samples, but only changing the eyepiece (new classifier) for your specific observation.

### 2. Fine-tuning

*   **Process:**
    1.  Load a pre-trained model.
    2.  Remove its original classification head.
    3.  Add a new, custom classification head.
    4.  **Unfreeze** some or all of the layers in the convolutional base.
    5.  Train the **entire model (or unfrozen layers + new head)** on your dataset with a **very small learning rate**.
*   **When to use:**
    *   Your dataset is **larger**.
    *   Your dataset is **somewhat different** from the original dataset.
*   **Layer-wise Fine-tuning:** Often, early layers (generic features) are kept frozen, while later layers (more task-specific features) are unfrozen and fine-tuned.
*   **Learning Rate:** Use a very small learning rate to avoid "unlearning" the valuable pre-trained features too quickly.

---

## Data Augmentation

*   **Definition:** Artificially increasing the size and diversity of your training dataset by creating modified versions of existing images.
*   **Purpose:**
    *   Prevents overfitting.
    *   Improves model generalization.
    *   Makes the model robust to variations in real-world data.
*   **Common Techniques:**
    *   **Geometric:**
        *   `RandomFlip` (horizontal/vertical)
        *   `RandomRotation`
        *   `RandomZoom`
        *   `RandomCrop`
        *   `RandomTranslation` (shifting)
    *   **Color:**
        *   `RandomBrightness`
        *   `RandomContrast`
        *   `ColorJitter`

---

## Influential Modern CNN Architectures

*   **VGG (Visual Geometry Group):**
    *   **Key Idea:** Simplicity through stacking many small (3x3) convolutional layers.
    *   **Impact:** Demonstrated the importance of network depth.
*   **ResNet (Residual Networks):**
    *   **Key Idea:** Introduced "skip connections" (residual blocks) to allow gradients to flow more easily, enabling training of extremely deep networks (e.g., 152 layers).
    *   **Impact:** Solved the vanishing/exploding gradient problem in very deep networks.
*   **Inception (GoogleNet):**
    *   **Key Idea:** "Inception modules" perform multiple parallel convolutions with different filter sizes (1x1, 3x3, 5x5) and pooling operations, then concatenate their outputs.
    *   **Impact:** Efficiently captures features at multiple scales and reduces computational cost.
*   **MobileNet:**
    *   **Key Idea:** Uses "depthwise separable convolutions" to significantly reduce parameters and computations.
    *   **Impact:** Designed for efficient deployment on mobile and embedded devices.
