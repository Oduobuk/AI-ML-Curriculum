Assignment: CI Pipeline with GitHub Actions & Monitoring Strategy

This assignment has two parts. In Part 1, you will create a practical Continuous Integration (CI) pipeline using GitHub Actions. In Part 2, you will design a monitoring strategy for a deployed model.

---

 Part 1: Building a CI Pipeline with GitHub Actions

 Objective
Your goal is to create a GitHub Actions workflow that automatically tests and lints a simple Python project every time you push a commit. This is the foundation of any MLOps pipeline.

 Setup
1.  Create a new public repository on GitHub.
2.  In that repository, create the following files:

    `project/operations.py`
    ```python
     A simple function to be tested
    def add(a, b):
        return a + b

    def subtract(a, b):
        return a - b
    ```

    `tests/test_operations.py`
    ```python
     Unit tests for the functions
    import sys
    sys.path.insert(0, 'project')
    from operations import add, subtract

    def test_add():
        assert add(2, 3) == 5
        assert add(-1, 1) == 0

    def test_subtract():
        assert subtract(5, 3) == 2
    ```

    `requirements.txt`
    ```
    pytest
    flake8
    ```

 Your Task
Create the GitHub Actions workflow file.
1.  Inside your repository, create the directory structure `.github/workflows/`.
2.  Inside that directory, create a new file named `ci-pipeline.yml`.
3.  Write a workflow in this file that does the following:
    - Triggers on every `push` to the `main` branch.
    - Defines a single job named `build-and-test`.
    - The job runs on `ubuntu-latest`.
    - The job has the following steps in order:
        1.  Checks out your repository's code.
        2.  Sets up a Python 3.8 environment.
        3.  Installs the dependencies from your `requirements.txt` file.
        4.  Runs `flake8` to lint the code in the `project/` directory.
        5.  Runs `pytest` to execute the unit tests in the `tests/` directory.

 Submission (Part 1)
- Push all your files (`operations.py`, `test_operations.py`, `requirements.txt`, and `ci-pipeline.yml`) to your GitHub repository.
- Go to the "Actions" tab in your repository and confirm that your workflow ran successfully.
- Submit the URL to your GitHub repository.

---

 Part 2: Designing a Model Monitoring Strategy

 Objective
For this part, you will act as an MLOps engineer and design a monitoring plan for the containerized Iris classifier you designed in last week's assignment.

 Your Task
Write a brief report (in a Markdown file named `monitoring_strategy.md`) that answers the following questions. Be specific.

1.  Key Performance Metric (KPI):
    - What is the primary performance metric you would monitor for this multi-class classification model (e.g., overall accuracy, macro F1-score, etc.)?
    - Why did you choose this metric?

2.  Data Drift Monitoring:
    - Which of the four input features (`sepal_length`, `sepal_width`, `petal_length`, `petal_width`) do you think is most important to monitor for drift?
    - What specific statistical test or metric (e.g., K-S test, PSI) would you use to monitor this feature's distribution?

3.  Prediction Drift Monitoring:
    - How would you monitor the distribution of the model's output predictions?
    - What would it signify if, over the course of a week, the model went from predicting classes `[0, 1, 2]` in a `[33%, 33%, 33%]` ratio to a `[10%, 10%, 80%]` ratio? What could be a possible real-world cause for this?

4.  Alerting Rule:
    - Define one critical alert you would implement.
    - Specify the metric it would track, the threshold that would trigger the alert, and the message the alert would send to your team. (e.g., "ALERT: Model accuracy has dropped below 90% for the last hour!").

 Submission (Part 2)
- Add the `monitoring_strategy.md` file to the same GitHub repository you created for Part 1.

 Grading Criteria
- Part 1 - GitHub Actions (60%): The workflow is correctly defined, runs successfully on a push, and correctly executes the linting and testing steps.
- Part 2 - Monitoring Strategy (40%): The monitoring report is well-reasoned, specific, and demonstrates a clear understanding of the key monitoring concepts (performance, data drift, prediction drift, and alerting).