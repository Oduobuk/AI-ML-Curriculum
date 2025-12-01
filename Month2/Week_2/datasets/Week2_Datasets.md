# Month 2, Week 2: Datasets for Classification (Logistic Regression & KNN)

This week, as we delve into classification algorithms like Logistic Regression and K-Nearest Neighbors (KNN), we need datasets that have discrete, categorical target variables. These datasets allow us to practice building models that predict categories rather than continuous values.

Here are some classic and commonly used datasets suitable for practicing classification:

## 1. Iris Dataset

*   **Description:** One of the most famous datasets in machine learning, often used for introductory classification tasks. It contains measurements of sepal length, sepal width, petal length, and petal width for 150 iris flowers.
*   **Features:** 4 numerical features (sepal length, sepal width, petal length, petal width).
*   **Target:** 3 classes of iris species: Setosa, Versicolor, and Virginica.
*   **Use Case:** Excellent for practicing multi-class classification with KNN and Logistic Regression (especially if you convert it to a binary problem by selecting two classes). It's clean and easy to visualize.

## 2. Titanic Dataset

*   **Description:** Contains data about passengers on the Titanic, including demographic information, ticket details, and whether they survived the disaster.
*   **Features:** A mix of numerical (age, fare) and categorical (sex, Pclass, embarked) features.
*   **Target:** Binary classification: survived (1) or not survived (0).
*   **Use Case:** A popular dataset for practicing binary classification, data cleaning, feature engineering (handling missing values, converting categorical to numerical), and understanding the impact of different features on survival.

## 3. Breast Cancer Wisconsin (Diagnostic) Dataset

*   **Description:** This dataset contains features computed from digitized images of a fine needle aspirate (FNA) of a breast mass. It describes characteristics of the cell nuclei present in the image.
*   **Features:** 30 numerical features (e.g., radius, texture, perimeter, area, smoothness, compactness, concavity, symmetry, fractal dimension, mean, standard error, and "worst" or largest values of these features).
*   **Target:** Binary classification: malignant (0) or benign (1).
*   **Use Case:** Ideal for binary classification tasks, especially for evaluating model performance in a medical context where false negatives can be critical. It's a good dataset for practicing feature scaling.

## 4. Wine Dataset

*   **Description:** Results of a chemical analysis of wines grown in the same region in Italy but derived from three different cultivars. The analysis determined the quantities of 13 constituents found in each of the three types of wines.
*   **Features:** 13 numerical features (e.g., alcohol, malic acid, ash, alkalinity of ash, magnesium, total phenols, flavanoids, nonflavanoid phenols, proanthocyanins, color intensity, hue, OD280/OD315 of diluted wines, proline).
*   **Target:** 3 classes of wine cultivars.
*   **Use Case:** Another good dataset for multi-class classification, allowing exploration of feature importance and different classification boundaries.

## 5. Credit Card Fraud Detection Dataset

*   **Description:** Contains anonymized credit card transactions labeled as fraudulent or legitimate. This dataset is highly imbalanced, with a very small percentage of fraudulent transactions.
*   **Features:** 28 anonymized numerical features (V1-V28) obtained via PCA, plus 'Time' and 'Amount'.
*   **Target:** Binary classification: fraud (1) or not fraud (0).
*   **Use Case:** A challenging real-world dataset for binary classification, particularly useful for understanding and addressing class imbalance, and for evaluating metrics beyond simple accuracy (e.g., precision, recall for the minority class).

These datasets are readily available through libraries like `scikit-learn` or can be found on platforms like Kaggle, making them easy to load and experiment with.
