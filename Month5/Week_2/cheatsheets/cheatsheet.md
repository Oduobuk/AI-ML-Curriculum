Cheatsheet: MLOps - Deployment and Containerization

 1. Containerization with Docker

Docker solves the "it works on my machine" problem by packaging an application and its dependencies into a consistent, portable container.

 Core Docker Concepts
| Concept | Description |
| :--- | :--- |
| Dockerfile | A text file with instructions to build a Docker image. It defines the base image, copies files, and installs dependencies. |
| Image | A read-only, lightweight, standalone, executable package that includes everything needed to run a piece of software: code, runtime, system tools, libraries, and settings. |
| Container | A runnable instance of an image. It's an isolated environment where the application runs. |

 Key Docker Commands
| Command | Purpose |
| :--- | :--- |
| `docker build -t <image_name> .` | Builds a Docker image from a `Dockerfile` in the current directory and tags it with a name. |
| `docker run -p 8000:80 <image_name>` | Runs a container from an image. The `-p` flag maps port 8000 on the host to port 80 in the container. |
| `docker ps` | Lists all running containers. |
| `docker images` | Lists all images on the local machine. |

---

 2. Creating an API for your ML Model

To make your model usable by other applications, you wrap it in a web API.

| Framework | Key Features | Best For |
| :--- | :--- | :--- |
| FastAPI | - High Performance (Asynchronous - ASGI)<br>- Automatic Docs (Swagger UI/ReDoc)<br>- Data Validation (via Pydantic type hints) | Production-grade APIs where performance, validation, and documentation are critical. |
| Flask | - Simple & Lightweight<br>- Flexible & Extensible<br>- Easy to learn | Quick prototypes, simple APIs, and projects where rapid development is prioritized over raw performance. |

---

 3. Cloud ML Platforms

Managed cloud services that streamline the entire machine learning lifecycle.

| Platform | Provider | Key Strengths & Focus |
| :--- | :--- | :--- |
| Amazon SageMaker | AWS | Flexibility & Scalability. Deep integration with the vast AWS ecosystem. Offers granular control over every step. |
| Vertex AI | Google Cloud | Ease of Use & AutoML. A unified, user-friendly platform that leverages Google's cutting-edge AI research and infrastructure (like TPUs). |
| Azure Machine Learning | Microsoft Azure | Enterprise Governance & Low-Code. Strong focus on security, compliance, and responsible AI. Features a visual designer for building models with minimal code. |