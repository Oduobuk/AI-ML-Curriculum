# Month 3, Week 1: Datasets for Clustering

This week, as we delve into unsupervised learning with clustering algorithms, we'll work with datasets where the goal is to discover inherent groupings or structures without prior knowledge of labels. These datasets are crucial for practicing how to apply clustering techniques and evaluate their effectiveness.

Here are some common types of datasets suitable for practicing clustering:

## 1. Synthetic Datasets (e.g., `make_blobs`, `make_moons`, `make_circles`)

*   **Description:** These datasets are generated programmatically using `scikit-learn` and are ideal for understanding and visualizing clustering algorithms. You can control parameters like the number of samples, number of clusters, standard deviation of clusters, and noise.
    *   `make_blobs`: Generates isotropic Gaussian blobs for clustering.
    *   `make_moons`: Generates two interleaving half-circles.
    *   `make_circles`: Generates a larger circle containing a smaller circle.
*   **Features:** Typically 2 or 3 numerical features, making them easy to plot.
*   **Target (for generation, but ignored for clustering):** Known cluster labels, which can be used to evaluate how well your algorithm rediscovered the true clusters.
*   **Use Case:** Perfect for initial experimentation, visualizing algorithm behavior, and understanding the strengths and weaknesses of different clustering methods (e.g., K-Means struggles with `make_moons` and `make_circles`).

## 2. Iris Dataset (Unlabeled)

*   **Description:** (As described in previous weeks) Contains measurements of sepal length, sepal width, petal length, and petal width for 150 iris flowers.
*   **Features:** 4 numerical features.
*   **Target (Ignored for clustering):** 3 classes of iris species.
*   **Use Case:** A classic dataset for clustering. By discarding the original species labels, you can apply clustering algorithms and then compare the discovered clusters to the true species labels to evaluate performance. It's good for demonstrating how clustering can reveal underlying structure.

## 3. Customer Segmentation Data

*   **Description:** Hypothetical or real-world data containing various attributes of customers, such as age, income, spending habits, purchase frequency, product preferences, etc.
*   **Features:** A mix of numerical and categorical features.
*   **Target:** None (unlabeled). The goal is to discover customer segments.
*   **Use Case:** A prime example of a business application for clustering. It allows you to practice identifying distinct customer groups for targeted marketing or product development. Requires careful feature engineering for categorical data.

## 4. Mall Customer Segmentation Data

*   **Description:** A publicly available dataset containing customer IDs, gender, age, annual income, and spending score (1-100) for mall customers.
*   **Features:** Age, Annual Income, Spending Score (and potentially Gender after encoding).
*   **Target:** None.
*   **Use Case:** A practical dataset for K-Means clustering to identify customer segments based on income and spending behavior. It's relatively clean and easy to work with.

## 5. Wholesale Customer Data

*   **Description:** Contains the annual spending (in monetary units) of clients on various product categories (e.g., Fresh, Milk, Grocery, Frozen, Detergents_Paper, Delicatessen) for a wholesale distributor. It also includes 'Channel' (Hotel/Restaurant/Cafe or Retail) and 'Region'.
*   **Features:** Annual spending on different product categories.
*   **Target:** None.
*   **Use Case:** Good for practicing clustering to identify different types of wholesale customers based on their purchasing patterns. You can also explore if the 'Channel' or 'Region' features align with the discovered clusters.

## 6. Seeds Dataset

*   **Description:** Contains measurements of geometric properties of kernels belonging to three different varieties of wheat.
*   **Features:** 7 numerical features (e.g., area, perimeter, compactness, length of kernel, width of kernel, asymmetry coefficient, length of kernel groove).
*   **Target (Ignored for clustering):** 3 varieties of wheat.
*   **Use Case:** Another clean, small dataset for multi-class clustering, similar to Iris, allowing for easy evaluation against known labels.

These datasets offer a range of complexities and characteristics, providing ample opportunities to apply and evaluate various clustering algorithms.
