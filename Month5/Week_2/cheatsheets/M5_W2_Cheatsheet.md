# Month 5, Week 2: MLOps Cheatsheet

## MLOps Principles

| Principle | Description |
| :--- | :--- |
| **Automation** | Automate every step of the ML lifecycle, from data ingestion to model deployment. |
| **CI/CD/CT** | Continuous Integration (CI), Continuous Delivery (CD), and Continuous Training (CT) adapted for ML. |
| **Versioning** | Version control for data, code, and models to ensure reproducibility. |
| **Monitoring** | Continuously monitor model performance, data drift, and system health in production. |
| **Collaboration** | Foster collaboration between data scientists, ML engineers, and operations teams. |
| **Governance** | Establish processes for model review, approval, and compliance. |

## The MLOps Lifecycle

1.  **Data Ingestion & Preparation:** Building robust and automated data pipelines.
2.  **Model Training & Tuning:** Experiment tracking, hyperparameter tuning, and model versioning.
3.  **Model Validation & Testing:** Evaluating model performance on unseen data and testing for fairness, bias, and security.
4.  **Model Packaging & Registry:** Packaging the model and its dependencies and storing it in a central model registry.
5.  **Model Deployment & Serving:** Deploying the model to a production environment for inference.
6.  **Model Monitoring & Observability:** Tracking model performance and detecting issues in production.
7.  **Model Retraining:** Automatically retraining and redeploying models as new data becomes available or performance degrades.

## Key Tools & Technologies

| Category | Tools |
| :--- | :--- |
| **Data Versioning** | DVC, Pachyderm, Dolt |
| **Experiment Tracking** | MLflow, Weights & Biases, Comet, Neptune |
| **Pipeline Orchestration** | Kubeflow Pipelines, Apache Airflow, TFX |
| **Model Registry** | MLflow Model Registry, AWS SageMaker Model Registry, Google AI Platform Models |
| **Model Serving** | TensorFlow Serving, TorchServe, Seldon Core, KServe (formerly KFServing) |
| **Monitoring** | Prometheus, Grafana, Evidently AI, Fiddler |
| **CI/CD** | GitHub Actions, GitLab CI/CD, Jenkins |
| **Infrastructure** | Docker, Kubernetes, AWS, Google Cloud, Azure |

## Deployment Strategies

| Strategy | Description | Use Case |
| :--- | :--- | :--- |
| **Batch Inference** | Predictions are generated periodically on a batch of data. | Non-real-time applications like product recommendations, customer segmentation. |
| **Real-time Inference** | Predictions are generated on-demand as new data arrives. | Real-time applications like fraud detection, dynamic pricing. |
| **Edge Deployment** | The model is deployed directly on an edge device (e.g., smartphone, IoT device). | Applications requiring low latency and offline capabilities. |
| **Canary Deployment** | The new model version is rolled out to a small subset of users before a full rollout. | Minimizing the risk of deploying a faulty model. |
| **A/B Testing** | Multiple model versions are deployed simultaneously to different user groups to compare their performance. | Selecting the best-performing model based on real-world metrics. |
