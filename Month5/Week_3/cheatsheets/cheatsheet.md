Cheatsheet: CI/CD and Model Monitoring

 1. The MLOps CI/CD Pipeline

A pipeline to automate the building, testing, and deployment of machine learning models.

 CI (Continuous Integration) Phase
Triggered by a code or data change.
1.  Test Code: Run `pytest` and `flake8` on the source code.
2.  Validate Data: Check schema and statistical properties of new data.
3.  Train & Evaluate Model: Run the training script and check if model performance (e.g., accuracy) meets a minimum threshold.
4.  Package Artifacts: Package the model, code, and dependencies into a reproducible format (usually a Docker container).

 CD (Continuous Deployment) Phase
Triggered by a successful CI run.
1.  Push to Registry: Push the Docker image to a container registry (e.g., Docker Hub, AWS ECR).
2.  Deploy to Staging/Production: Deploy the container to a server or cloud service.
3.  Monitor: Observe the model's live performance.

 GitHub Actions
- What it is: An automation tool built into GitHub to create CI/CD workflows.
- Where it lives: Workflows are defined in YAML files in the `.github/workflows/` directory.
- Key YAML Keywords:
    - `name`: The name of your workflow.
    - `on`: The event that triggers the workflow (e.g., `push`, `pull_request`).
    - `jobs`: A workflow consists of one or more jobs.
    - `runs-on`: The type of machine to run the job on (e.g., `ubuntu-latest`).
    - `steps`: A sequence of tasks within a job.
        - `uses`: Use a pre-built action from the marketplace (e.g., `actions/checkout@v3`).
        - `run`: Execute a shell command.

---

 2. Model Monitoring Concepts

Models degrade in production. Monitoring helps you detect when and why.

| Drift Type | What Changes? | Core Idea | How to Detect |
| :--- | :--- | :--- | :--- |
| Data Drift | The statistical distribution of the input data (X) changes. | The model is seeing new, unfamiliar data. | Compare input data distributions (e.g., using K-S test, Chi-Squared test, or Population Stability Index - PSI). |
| Concept Drift | The relationship between the input data (X) and the target variable (Y) changes. | The "rules of the game" have changed; learned patterns are no longer valid. | Track the model's performance metrics (e.g., accuracy, F1-score, error rate) over time. A sustained drop indicates drift. |

What to Monitor:
- Model Performance: Accuracy, precision, recall, F1-score, etc.
- Data Drift: Are the input features changing?
- Prediction Drift: Is the distribution of the model's output predictions changing?
- Operational Health: Latency, traffic, error rate of the API.

---

 3. Open-Source Monitoring Stack

A popular combination of tools to create a comprehensive monitoring dashboard.

| Tool | Role | Key Function |
| :--- | :--- | :--- |
| Evidently AI | The ML Specialist | A Python library that calculates ML-specific metrics like data drift, concept drift, and model performance by comparing production data to a reference dataset. |
| Prometheus | The Time-Series Database | Collects and stores the metrics exposed by the Evidently AI service over time. It "scrapes" an HTTP endpoint for new data points. |
| Grafana | The Dashboard | Visualizes the metrics stored in Prometheus. You can build interactive dashboards with graphs and tables, and set up alerts to be notified when a metric crosses a threshold. |

Workflow:
`Evidently AI calculates metrics` -> `Prometheus scrapes & stores metrics` -> `Grafana visualizes metrics & sends alerts`