# Month 4, Week 2 Exercises: Advanced CV & Transfer Learning

## Objective

These exercises are designed to deepen your understanding of transfer learning strategies, data augmentation, and the practical considerations when applying pre-trained models to new computer vision tasks.

---

### Exercise 1: Choosing the Right Strategy

For each scenario below, determine whether **Feature Extraction** or **Fine-tuning** would be the more appropriate transfer learning strategy. Justify your choice in 1-2 sentences.

1.  **Scenario A:** You are building a model to classify different types of fruits (apples, bananas, oranges, etc.). You have a very small dataset (50 images per fruit type) and the images are very similar to typical everyday objects found in ImageNet.
    *   **Strategy:**
    *   **Justification:**

2.  **Scenario B:** You are building a model to detect rare medical conditions from X-ray images. You have a moderately sized dataset (500 images per condition) and X-ray images look significantly different from natural images in ImageNet.
    *   **Strategy:**
    *   **Justification:**

3.  **Scenario C:** You are building a model to classify different species of insects. You have a large dataset (thousands of images per species) and while insects are natural objects, the fine-grained distinctions might not be perfectly captured by ImageNet's broader categories.
    *   **Strategy:**
    *   **Justification:**

---

### Exercise 2: Data Augmentation for Specific Problems

For each image classification problem, suggest 3-4 relevant data augmentation techniques and explain why each would be beneficial.

1.  **Problem:** Classifying different types of handwritten digits (like MNIST).
    *   **Techniques & Justification:**

2.  **Problem:** Detecting defects on manufactured circuit boards (images are taken under controlled lighting, but defects can appear at various angles).
    *   **Techniques & Justification:**

3.  **Problem:** Identifying different breeds of dogs from user-submitted photos (images can vary widely in lighting, background, and dog pose).
    *   **Techniques & Justification:**

---

### Exercise 3: Understanding Pre-trained Model Layers

Consider a pre-trained VGG16 model.

1.  Which layers (early, middle, or late convolutional layers) are generally considered to learn more "generic" features (e.g., edges, textures)?
2.  Which layers are generally considered to learn more "specific" features (e.g., object parts, complex patterns)?
3.  If you were fine-tuning VGG16 for a new task, and your dataset was very small but somewhat different from ImageNet, which layers would you be most inclined to unfreeze and fine-tune, and why?

---

### Exercise 4: Computational Benefits

Briefly explain (1-2 sentences) how transfer learning helps reduce the computational resources and time required for training a deep learning model compared to training from scratch.
