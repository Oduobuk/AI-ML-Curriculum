# Month 5, Week 2: MLOps & Production ML Assignment

## Instructions
This assignment is designed to give you hands-on experience with some of the core concepts of MLOps. You will be creating a simple, simulated CI/CD pipeline for a machine learning model using GitHub Actions.

## Task: Create a CI/CD Pipeline for a Dummy Model

**Objective:** To build a GitHub Actions workflow that automatically trains, evaluates, and "deploys" a simple machine learning model whenever new code is pushed to the repository.

**Model:**
*   Use the Iris dataset (available in scikit-learn).
*   Train a simple Logistic Regression model.

**Pipeline Requirements:**

1.  **Repository Setup:**
    *   Create a new public GitHub repository for this assignment.
    *   The repository should contain the following files:
        *   `train.py`: A Python script to train the model and save it to a file (e.g., `model.joblib`).
        *   `requirements.txt`: A file listing the necessary Python packages (e.g., scikit-learn, joblib).
        *   `.github/workflows/main.yml`: The GitHub Actions workflow file.

2.  **`train.py` Script:**
    *   The script should:
        *   Load the Iris dataset.
        *   Split the data into training and testing sets.
        *   Train a Logistic Regression model.
        *   Evaluate the model on the test set and print the accuracy.
        *   Save the trained model to a file.

3.  **GitHub Actions Workflow (`main.yml`):**
    *   The workflow should be triggered on every `push` to the `main` branch.
    *   It should consist of the following jobs:
        *   **`build`:**
            *   Checks out the code.
            *   Sets up a Python environment.
            *   Installs the dependencies from `requirements.txt`.
            *   Runs the `train.py` script.
        *   **`deploy` (Simulated):**
            *   This job should run only if the `build` job is successful.
            *   It should simulate a deployment by printing a message like "Deploying model to production..."
            *   **Bonus:** Modify the workflow so that this job only runs if the model accuracy from the `build` job is above a certain threshold (e.g., 90%). You can achieve this by saving the accuracy to a file in the `build` job and passing it as an artifact to the `deploy` job.

## Submission
*   Submit the link to your public GitHub repository containing the completed assignment.
*   Include a `README.md` file in your repository that explains your workflow and how to run it.
