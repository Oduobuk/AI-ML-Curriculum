# Month 3, Week 2: Datasets for Dimensionality Reduction

This week, as we explore Dimensionality Reduction techniques like PCA, t-SNE, and UMAP, we'll work with datasets that have a high number of features. The goal is to reduce this complexity while retaining meaningful information, often for visualization or to improve the performance of subsequent machine learning models.

Here are some common types of datasets suitable for practicing dimensionality reduction:

## 1. MNIST Handwritten Digits Dataset

*   **Description:** A large database of handwritten digits (0-9). It consists of 60,000 training images and 10,000 testing images. Each image is a 28x28 pixel grayscale image.
*   **Features:** 784 numerical features (pixel values).
*   **Target:** 10 classes (digits 0-9).
*   **Use Case:** A classic benchmark for dimensionality reduction. Reducing 784 dimensions to 2 or 3 allows for stunning visualizations of how well different algorithms separate the digits. PCA will show global structure, while t-SNE and UMAP will highlight local clusters.

## 2. Fashion MNIST Dataset

*   **Description:** A dataset of Zalando's article images, consisting of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes of clothing items.
*   **Features:** 784 numerical features (pixel values).
*   **Target:** 10 classes (e.g., T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot).
*   **Use Case:** Similar to MNIST but often considered more challenging. It's great for comparing how dimensionality reduction techniques handle more complex visual patterns and subtle differences between classes.

## 3. Olivetti Faces Dataset

*   **Description:** A dataset of face images. It contains 400 images of 40 distinct subjects (10 images each). The images are grayscale and 64x64 pixels.
*   **Features:** 4096 numerical features (pixel values).
*   **Target:** 40 classes (individuals).
*   **Use Case:** Excellent for demonstrating PCA for facial recognition (eigenfaces) and visualizing how different individuals cluster in a lower-dimensional space. It highlights the power of extracting principal components for image data.

## 4. Wine Dataset (Revisited)

*   **Description:** (As described in Week 2, Month 2) Results of a chemical analysis of wines grown in the same region in Italy but derived from three different cultivars.
*   **Features:** 13 numerical features.
*   **Target:** 3 classes of wine cultivars.
*   **Use Case:** A smaller dataset, but good for demonstrating PCA's ability to reduce the number of features while still maintaining class separability. It's a good starting point before moving to larger, more complex datasets.

## 5. Gene Expression Datasets

*   **Description:** These datasets typically contain measurements of gene activity (expression levels) across thousands of genes for various biological samples (e.g., different cell types, disease states).
*   **Features:** Thousands of numerical features (gene expression levels).
*   **Target (often ignored for unsupervised DR):** Biological categories (e.g., cancer types, cell lineages).
*   **Use Case:** A classic application area for dimensionality reduction. PCA is often used to identify major sources of variation, while t-SNE and UMAP are invaluable for visualizing complex relationships and identifying distinct cell populations.

## 6. ISOMAP / LLE (Manifold Learning) Datasets

*   **Description:** Synthetic datasets like the "Swiss Roll" or "S-curve" are often used to demonstrate manifold learning algorithms (which UMAP is related to). These datasets are intrinsically low-dimensional but embedded in a higher-dimensional space.
*   **Features:** 3 numerical features (for Swiss Roll).
*   **Target:** None (unlabeled).
*   **Use Case:** While not directly for PCA/t-SNE/UMAP, these datasets are excellent for understanding the concept of a "manifold" and how non-linear dimensionality reduction techniques can "unroll" such structures.

These datasets provide a range of complexities and characteristics, offering ample opportunities to apply and evaluate various dimensionality reduction algorithms.