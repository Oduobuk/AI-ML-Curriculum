Lecture 2: Model Monitoring and Observability

Deploying a model is not the end of the machine learning lifecycle. Once in production, a model's performance can and will degrade over time. Model Monitoring is the process of tracking a model's performance in a live environment to detect and diagnose issues.

 Why Models Degrade: The Problem of Drift

The world is not static. The data your model sees in production will eventually differ from the data it was trained on. This phenomenon is called drift, and it's the primary reason for performance degradation. There are two main types of drift.

 1. Data Drift (or Covariate Shift)

Data drift occurs when the statistical properties of the input data (X) change. The relationship between the inputs and the output remains the same, but the model is seeing new or different kinds of data that it wasn't prepared for.

- What Changes: The distribution of `P(X)`.
- Core Idea: The world the model is seeing has changed.
- Example: A loan approval model is trained on data from a specific country. It is then deployed in a new country where the average income, age, and credit history distributions are different. The model's performance will likely suffer because it is operating on a demographic it knows little about.
- Detection: Data drift is detected by comparing the distribution of incoming production data with a baseline (usually the training data). This is done using statistical tests like the Kolmogorov-Smirnov (K-S) test for numerical features or the Chi-Squared test for categorical features. Metrics like Population Stability Index (PSI) are also common.

 2. Concept Drift

Concept drift is more subtle. It occurs when the relationship between the input features (X) and the target variable (Y) changes over time. The patterns the model learned to make predictions are no longer valid.

- What Changes: The conditional probability `P(Y|X)`.
- Core Idea: The meaning of the data has evolved.
- Example: A spam detection model was trained when spammers used keywords like "free money". Over time, spammers adapt and start using more sophisticated image-based spam. The input data (emails) might still have similar word distributions, but the concept of what constitutes spam has changed, making the original model obsolete.
- Detection: Since concept drift directly impacts the model's predictive power, the most direct way to detect it is by monitoring the model's performance metrics (e.g., accuracy, F1-score, precision, recall). A significant, sustained drop in performance is a strong indicator of concept drift.

| Feature | Data Drift | Concept Drift |
| :--- | :--- | :--- |
| What Changes? | The distribution of the input data P(X). | The relationship between inputs and the output P(Y|X). |
| The Problem | The model sees unfamiliar data. | The learned patterns are no longer correct. |
| Detection Method | Statistical comparison of data distributions. | Monitoring of model performance metrics. |

 What to Monitor: The Pillars of ML Observability

A robust monitoring strategy goes beyond just drift and looks at the entire system.

1.  Model Performance:
    - Business Metrics: How is the model impacting key business KPIs? (e.g., click-through rate, revenue).
    - Evaluation Metrics: Track standard ML metrics like accuracy, precision, recall, F1-score, MAE, etc. This requires having access to ground truth labels, which may be delayed.

2.  Data Quality & Drift:
    - Data Schema: Are you receiving the features you expect? Are their data types correct?
    - Data Drift: Is the distribution of incoming data changing compared to the training data?
    - Outliers: Is the model receiving a significant number of outlier values it wasn't trained on?

3.  Prediction Drift:
    - If ground truth is delayed, you can monitor the distribution of the model's predictions. A significant shift (e.g., a fraud model suddenly starts predicting "fraud" 50% of the time instead of 1%) can be an early warning sign of a problem.

4.  Operational Health:
    - Latency: How long does it take for the model to return a prediction?
    - Traffic: How many requests is the model serving?
    - Error Rate: What percentage of requests are failing due to system errors?

 Responding to Drift

Detecting drift is only half the battle. The goal of monitoring is to trigger an action. When drift is detected, the typical response is to retrain the model on a new, more representative dataset that includes the recent production data. This is the core idea behind Continuous Training (CT).