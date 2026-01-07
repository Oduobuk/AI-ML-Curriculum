 Lecture 3: Advanced CI/CD and Branching Strategies

As machine learning projects grow in complexity and team size, having a disciplined approach to managing code and experiments becomes critical. This lecture explores how to adapt a popular software development branching strategy, GitFlow, for the unique needs of ML projects.

---

 1. The Need for a Branching Strategy

In any software project, a branching strategy is a set of rules for how a team uses Git branches. It answers questions like:
- Where does new work happen?
- How do we prepare for a new release?
- How do we fix urgent bugs in production?

Without a strategy, a project's Git history can become a tangled mess, making it difficult to track changes, collaborate, and maintain a stable production version. For ML, this is even more critical because we need to track not just code, but also experiments.

---

 2. GitFlow: A Structured Branching Model

GitFlow is a robust branching model that uses two long-lived branches and several types of temporary, supporting branches.

 Long-Lived Branches
1.  `main` (or `master`): This branch is the source of truth for production. It should always be stable and deployable. Code is never committed directly to `main`.
2.  `develop`: This is the primary development branch. All new features and experiments that have been completed and approved are merged into `develop`. It represents the "next release."

 Supporting Branches
1.  `feature/`: When you start working on a new feature, you create a branch from `develop` (e.g., `feature/add-user-authentication`). Once the feature is complete, it is merged back into `develop`.
2.  `release/`: When the `develop` branch has enough features for a new release, a `release` branch is created from `develop` (e.g., `release/v1.2.0`). This branch is used for final testing, bug fixes, and preparing for the release. No new features are added here. Once ready, the `release` branch is merged into both `main` and `develop`.
3.  `hotfix/`: If a critical bug is discovered in production (`main`), a `hotfix` branch is created directly from `main` (e.g., `hotfix/fix-login-bug`). The fix is made here and then merged into both `main` and `develop` to ensure the fix is included in the next release.

![GitFlow Diagram](https://wac-cdn.atlassian.com/dam/jcr:61ccc62a-5269-43fb-a71f-a4b53f6f5340/05.svg?cdnVersion=1533)
Image source: Atlassian

---

 3. Adapting GitFlow for Machine Learning

The standard GitFlow model needs adaptation to handle the highly experimental nature of machine learning. The key is to rethink the `feature` branch.

 The `experiment/` Branch
Instead of `feature/`, it's more intuitive to use experiment/ branches for ML projects. Each `experiment` branch represents a specific hypothesis you are testing.

- Creating an Experiment: When you want to try a new model, a different set of hyperparameters, or a new data preprocessing technique, you create a branch from `develop`.
  ```bash
   Start a new experiment from the develop branch
  git checkout develop
  git checkout -b experiment/try-resnet-model
  ```
- Isolating Work: All code changes, configuration updates, and notebook explorations for this experiment happen on this branch. This keeps the `develop` branch clean and stable.
- Tracking with Tools: While working on an `experiment` branch, you should use tools like MLflow or DVC to track your work.
    - DVC: If you change the dataset or feature engineering process, use `dvc add` to version the data.
    - MLflow: Log your parameters, metrics, and model artifacts for every run within the experiment.
- Merging a Successful Experiment: If an experiment is successful (e.g., it produces a model with better performance), you create a pull request to merge the `experiment/try-resnet-model` branch back into `develop`. This integrates the improved code into the main development line.
- Abandoning a Failed Experiment: If an experiment fails, you can simply abandon the branch. No harm is done to the `develop` or `main` branches.

 The Full ML Workflow with Adapted GitFlow

1.  Start New Work: A data scientist creates an `experiment/new-idea` branch from `develop`.
2.  Iterate: The data scientist works on the branch, committing code changes and using DVC/MLflow to track experiments.
3.  CI Pipeline Runs: Every push to the `experiment` branch can trigger a CI pipeline (via GitHub Actions) that runs linting, unit tests, and maybe even a short model training run to ensure nothing is broken.
4.  Review and Merge: Once the experiment yields a successful model, a pull request is opened. The team reviews the code and the experiment results (e.g., via a link to an MLflow report). Upon approval, the branch is merged into `develop`.
5.  Prepare for Release: When enough successful experiments have been merged into `develop`, a `release/v2.1.0` branch is created. On this branch, final integration testing and model validation occur.
6.  Deploy: Once the `release` branch is stable, it's merged into `main`. This merge triggers the final CD pipeline, which deploys the new model to production. The `release` branch is also merged back into `develop`.
7.  Hotfix: If the production model has a critical bug, a `hotfix` branch is created from `main`, fixed, and merged back into both `main` and `develop`.

This adapted GitFlow provides a structured, scalable, and reproducible framework for managing complex team-based ML projects.
