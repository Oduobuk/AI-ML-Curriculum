Assignment: Containerizing a Scikit-Learn Model with FastAPI and Docker

 Objective

The goal of this assignment is to take a pre-trained machine learning model, wrap it in a web API using FastAPI, and then containerize the entire application using Docker. This is a foundational MLOps workflow for deploying models in a portable and scalable way.

You will be provided with a simple, pre-trained Iris classification model from Scikit-learn.

 Provided Files

First, create the following two files.

1. `train_model.py`
This script trains a simple Logistic Regression model on the Iris dataset and saves it to a file named `iris_model.joblib`. You do not need to modify this script, just run it once to generate the model file.

```python
 train_model.py
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
import joblib

 Load data and train model
iris = load_iris()
X, y = iris.data, iris.target

model = LogisticRegression(max_iter=200)
model.fit(X, y)

 Save the trained model
joblib.dump(model, 'iris_model.joblib')

print("Model trained and saved as iris_model.joblib")
```

2. `iris_model.joblib`
Run the `train_model.py` script (`python train_model.py`). This will create the `iris_model.joblib` file in your project directory.

 Your Tasks

You need to create the following three files to complete the assignment.

 1. `requirements.txt`
Create a `requirements.txt` file that lists all the Python dependencies needed to run your application. At a minimum, you will need:
- `fastapi`
- `uvicorn` (an ASGI server to run FastAPI)
- `scikit-learn`
- `joblib`

 2. `main.py` (The FastAPI Application)
Create a Python script named `main.py` that does the following:
- Imports `FastAPI`, `BaseModel` from `pydantic`, and `joblib`.
- Loads the pre-trained model `iris_model.joblib`.
- Creates a `FastAPI` app instance.
- Defines a Pydantic `BaseModel` to represent the input data for a prediction. The Iris dataset has 4 features: sepal length, sepal width, petal length, and petal width. Your model should reflect this.
- Creates a POST endpoint (e.g., at `/predict`) that:
    - Accepts the input data based on your Pydantic model.
    - Uses the loaded model to make a prediction.
    - Returns the prediction in a JSON response. The Iris dataset has 3 classes (0, 1, 2), which correspond to the flower species.

 3. `Dockerfile`
Create a `Dockerfile` to containerize your FastAPI application. It should:
- Start from a `python:3.8-slim` base image.
- Set a working directory (e.g., `/app`).
- Copy your `requirements.txt`, `main.py`, and `iris_model.joblib` files into the container.
- Install the Python dependencies from `requirements.txt`.
- Expose the port your application will run on (FastAPI with Uvicorn defaults to port 8000).
- Define the command to run your application using `uvicorn`. The command should be `uvicorn main:app --host 0.0.0.0 --port 8000`.

 How to Test Your Solution
1.  Build the Docker image: `docker build -t iris-classifier .`
2.  Run the Docker container: `docker run -p 8000:8000 iris-classifier`
3.  Test the API: Open your web browser and go to `http://127.0.0.1:8000/docs`. You should see the automatic FastAPI documentation. Use the "Try it out" feature to send a POST request to your `/predict` endpoint with some sample data, for example:
    ```json
    {
      "sepal_length": 5.1,
      "sepal_width": 3.5,
      "petal_length": 1.4,
      "petal_width": 0.2
    }
    ```
    You should get a prediction back from your model.

 Submission
Submit the three files you created: `requirements.txt`, `main.py`, and `Dockerfile`.

 Grading Criteria
- `requirements.txt` (10%): Correctly lists all necessary dependencies.
- `main.py` (40%): Correctly loads the model and implements the FastAPI application with a working prediction endpoint and Pydantic data validation.
- `Dockerfile` (40%): Correctly builds a runnable container image for the application.
- Overall Functionality (10%): The container runs and the API endpoint returns predictions successfully.