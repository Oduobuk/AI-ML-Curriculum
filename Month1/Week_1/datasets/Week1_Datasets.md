# Week 1: Classic Datasets for Beginners

When you begin your journey in machine learning, you will frequently encounter a few "classic" datasets. These datasets are excellent for learning and practice because they are well-understood, relatively small, and clean.

They are built into many machine learning libraries like Scikit-learn, so you can often load them with a single line of code.

## 1. The Iris Dataset

*   **Description:** This is arguably the most famous dataset in pattern recognition. It contains measurements for 150 iris flowers from three different species.
*   **Task:** Classification. The goal is to predict the species of an iris flower based on its measurements.
*   **Features (Input):**
    *   Sepal Length (cm)
    *   Sepal Width (cm)
    *   Petal Length (cm)
    *   Petal Width (cm)
*   **Target (Output):**
    *   Species: Iris Setosa, Iris Versicolour, or Iris Virginica.
*   **Use Case:** Perfect for understanding basic classification algorithms.

## 2. The MNIST Dataset of Handwritten Digits

*   **Description:** A large database of handwritten digits that is commonly used for training and testing image processing systems. It contains 60,000 training images and 10,000 testing images.
*   **Task:** Image Classification. The goal is to identify the digit (0-9) from a grayscale image.
*   **Features (Input):**
    *   A 28x28 pixel grayscale image. Each pixel has a brightness value from 0 (black) to 255 (white). This is typically flattened into a 784-element array.
*   **Target (Output):**
    *   The digit (0, 1, 2, 3, 4, 5, 6, 7, 8, or 9).
*   **Use Case:** The "Hello, World!" of deep learning and computer vision. It's a great starting point for building your first neural network.

## 3. The Titanic Dataset

*   **Description:** This dataset contains information about passengers on the Titanic, including whether or not they survived the disaster.
*   **Task:** Binary Classification. The goal is to predict whether a passenger survived or not.
*   **Features (Input):**
    *   Passenger Class (Pclass)
    *   Sex
    *   Age
    *   Number of Siblings/Spouses Aboard (SibSp)
    *   Number of Parents/Children Aboard (Parch)
    *   Fare
    *   Port of Embarkation (where they got on)
*   **Target (Output):**
    *   Survived (0 = No, 1 = Yes)
*   **Use Case:** An excellent dataset for practicing data cleaning, feature engineering (creating new features from existing ones), and building a predictive classification model. It's a popular challenge on the Kaggle platform.
