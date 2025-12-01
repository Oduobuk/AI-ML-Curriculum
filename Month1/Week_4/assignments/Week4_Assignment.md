# Week 4 Assignment: Visualizing Model Performance

## Objective

This assignment will test your ability to use the `matplotlib` library to create visualizations that compare the performance of different machine learning model configurations.

## Scenario

Imagine you have trained three different models (or three versions of the same model with different hyperparameters) for a classification task. You have collected the following performance metrics:

| Model Configuration | Accuracy (%) | Inference Time (ms) |
|---------------------|--------------|-----------------------|
| Model A (Baseline)  | 85           | 50                    |
| Model B (Complex)   | 92           | 150                   |
| Model C (Simple)    | 78           | 25                    |

Your task is to create two different plots to visualize these results.

## Instructions

1.  **Create a new Python file:** Name it `visualize_performance.py`.

2.  **Import Matplotlib:** Make sure to import the library at the top of your script:
    ```python
    import matplotlib.pyplot as plt
    ```

3.  **Store the Data:** Store the model names, accuracies, and inference times in Python lists.

4.  **Part 1: Create a Bar Chart for Accuracy**
    *   Create a bar chart that compares the accuracy of the three models.
    *   **Requirements:**
        *   The Y-axis should represent the accuracy score.
        *   The X-axis should show the names of the three models.
        *   Give the chart a clear title (e.g., "Model Accuracy Comparison").
        *   Label the Y-axis (e.g., "Accuracy (%)").
        *   Save the plot to a file named `accuracy_comparison.png`.
        *   Display the plot.

5.  **Part 2: Create a Scatter Plot for Accuracy vs. Time**
    *   Create a scatter plot to visualize the trade-off between accuracy and inference time.
    *   **Requirements:**
        *   The X-axis should represent Inference Time (ms).
        *   The Y-axis should represent Accuracy (%).
        *   Each point on the plot should represent one of the models.
        *   Give the chart a clear title (e.g., "Model Performance: Accuracy vs. Inference Time").
        *   Label both the X-axis and the Y-axis.
        *   **Bonus:** Can you add text labels (`Model A`, `Model B`, `Model C`) next to each point on the scatter plot? (Hint: Use `plt.text()`).
        *   Save the plot to a file named `accuracy_vs_time.png`.
        *   Display the plot.

## Submission

*   Save your completed Python code in the `visualize_performance.py` file.
*   You should also have two new image files in your directory: `accuracy_comparison.png` and `accuracy_vs_time.png`.
