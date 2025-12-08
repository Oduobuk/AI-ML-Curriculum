# Month 5, Week 3: Explainable and Responsible AI Assignment

## Instructions
This assignment will give you hands-on experience with two popular XAI (Explainable AI) techniques: LIME and SHAP. You will train a model and then use these techniques to explain its predictions.

## Task: Explain a "Black Box" Model

**Objective:** To train a classifier, treat it as a black box, and then use LIME and SHAP to understand its behavior.

**Dataset:**
*   Use the **Breast Cancer Wisconsin (Diagnostic) Dataset**, which is available in scikit-learn. This is a classic binary classification dataset.

**Model:**
*   Train a **Gradient Boosting Classifier** (`sklearn.ensemble.GradientBoostingClassifier`). This model is powerful but can be difficult to interpret directly, making it a good candidate for XAI.

**Pipeline:**

1.  **Data Loading and Preparation:**
    *   Load the dataset from `sklearn.datasets`.
    *   Split the data into training and testing sets.

2.  **Model Training:**
    *   Train a `GradientBoostingClassifier` on the training data.
    *   Evaluate the model's accuracy on the test set.

3.  **Explainability with LIME:**
    *   Install the `lime` library (`pip install lime`).
    *   Create a LIME explainer object for your trained model.
    *   Choose **two** instances from your test set: one that was correctly classified and one that was misclassified (if any).
    *   For each instance, generate a LIME explanation and visualize it.
    *   In your submission, include the LIME plots and a brief interpretation of what they tell you about why the model made its prediction for each instance.

4.  **Explainability with SHAP:**
    *   Install the `shap` library (`pip install shap`).
    *   Create a SHAP explainer object for your model.
    *   Calculate the SHAP values for your test set.
    *   Generate the following SHAP plots:
        *   A **summary plot** (e.g., a bee swarm plot) to show the global feature importance.
        *   A **force plot** for one of the instances you explained with LIME.
    *   In your submission, include the SHAP plots and a brief interpretation of what they reveal about your model's behavior.

## Submission
*   A single Jupyter Notebook or Python script that contains all your code for training the model and generating the LIME and SHAP explanations.
*   Your notebook should include the generated plots and your written interpretations of the results.
