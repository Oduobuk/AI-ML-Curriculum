Lecture 1: CI/CD for Machine Learning with GitHub Actions

 What is CI/CD and Why is it Different for ML?

Continuous Integration (CI) and Continuous Deployment (CD) are practices from DevOps that automate the process of building, testing, and releasing software.

- CI: Automatically integrating code changes from multiple contributors into a single shared repository. Each integration is verified by an automated build and test sequence.
- CD: Automatically deploying all code changes that pass the CI stage to a testing or production environment.

For Machine Learning (a practice often called MLOps), CI/CD is more complex than for traditional software. Instead of just managing code, we must manage three components: code, models, and data.

This introduces unique challenges:
- Versioning: We need to version not just our code (with Git), but also our datasets and the trained models to ensure reproducibility.
- Complex Testing: Testing isn't just about code functionality. We also need to validate data, evaluate model performance, and check for fairness and bias.
- Continuous Training (CT): A new concept in MLOps, CT is the practice of automatically retraining models on new data to prevent performance degradation.

 A Typical MLOps CI/CD Pipeline

A mature CI/CD pipeline for an ML project automates the entire lifecycle.

 1. Continuous Integration (CI) Phase
This phase is triggered by a new code commit or the addition of new data.

1.  Source & Test Code: Run standard unit and integration tests on your Python code (e.g., using `pytest`). Linting (`flake8`) is also performed here.
2.  Validate Data: Automatically check new data for schema consistency, statistical properties, and quality. If the data is invalid, the pipeline fails.
3.  Train & Evaluate Model: The pipeline automatically trains the model on the data. The model's performance (e.g., accuracy, F1-score) is evaluated against a predefined threshold.
4.  Validate Model: The newly trained model is compared against the currently deployed model. It should only be promoted if it shows a significant performance improvement.
5.  Package Model: The validated model, along with its code and dependencies, is packaged into a reproducible format, most commonly a Docker container.

 2. Continuous Deployment (CD) Phase
Once the CI phase is successful, the packaged model is automatically deployed.

1.  Push to Registry: The Docker container is pushed to a container registry (like Docker Hub, AWS ECR, or Google Artifact Registry).
2.  Staged Deployment: To minimize risk, the model is deployed gradually.
    - Shadow Deployment: The new model runs alongside the old one, receiving live traffic but not sending its predictions back to the user. This allows for monitoring its live performance without risk.
    - Canary Release: The new model is rolled out to a small subset of users (e.g., 5%) before a full release.
3.  Promote or Rollback: If the new model performs well in the staged deployment, it is promoted to serve all traffic. If not, the system automatically rolls back to the previous version.

 Using GitHub Actions for MLOps

GitHub Actions is a powerful CI/CD platform built directly into GitHub. It allows you to automate workflows using simple YAML files stored in the `.github/workflows/` directory of your repository.

 Example GitHub Actions Workflow

This workflow defines a CI pipeline that is triggered on every push to the `main` branch. It tests the code, trains a model, and then could be extended to deploy it.

`.github/workflows/ci-pipeline.yml`
```yaml
name: ML Model CI Pipeline

 Trigger the workflow on a push to the main branch
on:
  push:
    branches: [ main ]

 Define the jobs to run
jobs:
  build-and-test:
    runs-on: ubuntu-latest  Use a GitHub-hosted Linux runner

    steps:
    1. Check out the repository code
    - name: Checkout code
      uses: actions/checkout@v3

    2. Set up a Python environment
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.8'

    3. Install dependencies
    - name' Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt

    4. Lint the code
    - name: Lint with flake8
      run: |
        pip install flake8
        flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics

    5. Run unit tests
    - name: Test with pytest
      run: |
        pip install pytest
        pytest

    6. (Placeholder) Train the model
     In a real pipeline, this step would run your training script
    - name: Train model
      run: echo "Running training script..."  Replace with 'python train.py'

    7. (Placeholder) Evaluate the model
    - name: Evaluate model
      run: echo "Running evaluation script..."  Replace with 'python evaluate.py'
```

 Key Components of the Workflow
- `on`: The event trigger. Here, it's a `push` to the `main` branch.
- `jobs`: A workflow is made of one or more jobs. Here, we have one job named `build-and-test`.
- `runs-on`: Specifies the type of machine to run the job on. `ubuntu-latest` is a common choice.
- `steps`: A sequence of tasks.
    - `uses:`: This keyword specifies that the step will use a pre-built action from the GitHub Marketplace (e.g., `actions/checkout@v3` to get the code).
    - `run:`: This keyword specifies that the step will run a shell command.