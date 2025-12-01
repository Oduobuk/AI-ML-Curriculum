# Week 4: Data for Visualization

This week focuses on **data visualization**, a crucial step in both exploring a dataset (Exploratory Data Analysis) and in evaluating the performance of a model.

The "data" we visualize can come from two main sources:

### 1. The Original Dataset

We often create plots to understand the dataset itself before we even start building models. This is part of **Exploratory Data Analysis (EDA)**.

For this, we can use the classic datasets introduced in Week 1:

*   **The Iris Dataset:**
    *   You could create a **scatter plot** of Petal Length vs. Petal Width and color-code the points by species. This would visually show you how separable the classes are.
    *   You could create **histograms** or **box plots** for each of the four features (Sepal Length, etc.) to see their distributions.

*   **The Titanic Dataset:**
    *   You could create a **bar chart** showing the number of survivors vs. non-survivors.
    *   You could create a **bar chart** showing survival rate by passenger class (Pclass), which would quickly reveal that first-class passengers were more likely to survive.

### 2. The Output of Your Model

This is what the lecture and assignment for this week focus on. After training a model, you generate data *about* the model's performance. This is the data you visualize to evaluate and compare models.

Examples of "model output data" include:

*   **Performance Metrics:** A list of accuracy scores, precision scores, or F1-scores for different models. This is perfect for a **bar chart**.
*   **Trade-off Metrics:** A set of paired metrics like (Accuracy, Inference Time) or (Quality, Speed). This is perfect for a **scatter plot** to visualize the Pareto frontier.
*   **Training History:** The value of the loss function at each epoch during training. This is perfect for a **line plot** to see if the model is learning over time.

For the exercises and assignment this week, you will be creating these small, summary datasets manually (e.g., by typing `accuracies = [0.85, 0.92, 0.78]`). In a real project, you would generate this data by running and testing your models.
