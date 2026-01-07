Exercises: CI/CD and Monitoring Concepts

 Exercise 1: Design a CI/CD Workflow

You have a machine learning project with the following files:
- `train.py`: A script to train a model.
- `evaluate.py`: A script to evaluate the model's performance.
- `requirements.txt`: A file with Python dependencies.
- `tests/`: A directory with unit tests for your data processing functions.

You want to create a GitHub Actions workflow that runs every time you push a commit to the `main` branch.

Task: List the key steps you would include in your `ci-workflow.yml` file. Put them in the correct logical order. You don't need to write the full YAML, just describe the steps (e.g., "1. Check out code", "2. Install Python", etc.).

 Exercise 2: Data Drift vs. Concept Drift

For each of the following scenarios, identify whether the problem is more likely due to Data Drift or Concept Drift. Explain your reasoning.

1.  Scenario A: A model that predicts house prices was trained on data from 2010-2019. After a major economic boom in 2021, the model consistently underestimates house prices, even for houses similar to those in the training set. The features (square footage, number of bedrooms) are being recorded in the same way.
2.  Scenario B: A sentiment analysis model for product reviews works well. The company expands its e-commerce site to a new country. The model's performance drops significantly on reviews from the new country, which contain different slang and cultural references.
3.  Scenario C: A fraud detection model was trained to identify fraudulent transactions based on features like transaction amount and location. A new type of fraud emerges where criminals make many small transactions from constantly changing locations, a pattern the model has never seen and which doesn't fit the learned rules of "fraud".
4.  Scenario D: A medical imaging model that detects a certain disease in X-rays performs well. A new hospital wing is opened with brand new X-ray machines that produce images with a slightly different contrast and brightness profile than the machines used to generate the training data. The model's accuracy begins to decline.

 Exercise 3: Designing a Monitoring Strategy

You have just deployed the Iris classifier model from last week's assignment. It's wrapped in a FastAPI and running in a Docker container.

Task: Describe a basic monitoring strategy for this model. Answer the following questions:

1.  Performance Metric: What is the single most important model performance metric you would track? (e.g., accuracy, F1-score, etc.)
2.  Data Drift: Which input feature (`sepal_length`, `sepal_width`, `petal_length`, `petal_width`) would you be most concerned about monitoring for data drift? Why?
3.  Prediction Drift: How would you monitor for prediction drift? What would it mean if your model, which used to predict the three flower classes in roughly equal numbers, suddenly started predicting class `2` (Virginica) 90% of the time?
4.  Alerting: Describe a specific alert you would set up. What metric would it watch, what would be the threshold, and what would the alert message say? (e.g., "Metric X has dropped below Y!").

 Exercise 4: The Monitoring Stack

Briefly describe the primary role of each of the following tools in an open-source ML monitoring stack:

1.  Evidently AI: What is its main job?
2.  Prometheus: What is its main job?
3.  Grafana: What is its main job?