Exercises: MLOps Concepts

 Exercise 1: Write a Dockerfile

You have a simple Python application with the following structure:

```
/my_ml_app
├── app.py
└── requirements.txt
```

- `app.py` is a Python script that needs to be run.
- `requirements.txt` contains the Python libraries the app depends on (e.g., `scikit-learn`, `flask`).

Task: Write a basic `Dockerfile` that accomplishes the following:
1.  Starts from a slim Python 3.8 base image (`python:3.8-slim`).
2.  Sets the working directory inside the container to `/app`.
3.  Copies all the files from your host's current directory into the container's working directory.
4.  Installs the dependencies listed in `requirements.txt`.
5.  Specifies that the command to run when the container starts is `python app.py`.

Fill in the blanks in the Dockerfile below:

```dockerfile
 Use an official Python runtime as a parent image
FROM ______

 Set the working directory
WORKDIR ______

 Copy the project files into the container
COPY . .

 Install dependencies
RUN pip install -r ______

 Command to run the application
CMD [______, ______]
```

 Exercise 2: Choosing the Right API Framework

For each of the following scenarios, choose whether FastAPI or Flask would be a better fit and briefly explain why.

1.  Scenario A: You are a data scientist who has just trained a new model. You need to create a quick, simple web demo for internal stakeholders by the end of the day.
2.  Scenario B: You are part of a team building a new, public-facing API for a machine learning service that is expected to handle thousands of concurrent requests. The API needs to be highly reliable and well-documented for external developers.
3.  Scenario C: You are building a service where the input data must strictly conform to a specific nested JSON structure. You want to automatically validate incoming requests and return clear errors if the format is wrong.

 Exercise 3: Choosing the Right Cloud Platform

Match the cloud ML platform to the company profile that would most likely choose it.

Platforms:
- A. Amazon SageMaker
- B. Google Vertex AI
- C. Azure Machine Learning

Company Profiles:
1.  A large financial services company that uses Microsoft Active Directory for user management and Power BI for analytics. They have strict data governance and compliance requirements.
2.  A fast-growing startup that is already using Google BigQuery for their data warehouse and wants to quickly build and deploy models using the latest AutoML technology.
3.  A large e-commerce company whose entire infrastructure, from databases to storage to web servers, is already running on AWS. Their expert ML team wants maximum flexibility and control to build custom training and deployment pipelines.

 Exercise 4: Docker Commands

What do the following Docker commands do?

1.  `docker build -t my-app:1.0 .`
2.  `docker run -p 5000:8080 my-app:1.0`
3.  `docker ps`