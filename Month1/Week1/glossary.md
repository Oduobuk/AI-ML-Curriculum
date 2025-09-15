# Week 1: Glossary of Terms

## Core AI Concepts

### Artificial Intelligence (AI)
**Definition**: The simulation of human intelligence processes by machines, especially computer systems. These processes include learning, reasoning, problem-solving, perception, and language understanding.
**Detailed Explanation**: AI systems are designed to perform tasks that typically require human intelligence, such as visual perception, speech recognition, decision-making, and language translation. AI can be categorized into Narrow AI (designed for specific tasks) and General AI (hypothetical AI with human-like cognitive abilities).
**Example**: Virtual assistants like Siri or Alexa that can understand and respond to voice commands.

### Machine Learning (ML)
**Definition**: A subset of AI that provides systems the ability to automatically learn and improve from experience without being explicitly programmed.
**Detailed Explanation**: ML focuses on the development of computer programs that can access data and use it to learn for themselves. The learning process begins with observations or data, such as examples, direct experience, or instruction, to look for patterns in data and make better decisions in the future.
**Example**: Email spam filters that learn to identify spam based on user actions.

### Deep Learning
**Definition**: A specialized form of machine learning that uses neural networks with many layers (deep neural networks) to analyze various factors in large amounts of data.
**Detailed Explanation**: Deep learning models are built using neural networks that have multiple hidden layers between the input and output layers. These models can automatically discover the representations needed for feature detection or classification from raw data.
**Example**: Image recognition systems that can identify objects in photos with high accuracy.

### Natural Language Processing (NLP)
**Definition**: A field of AI that enables computers to understand, interpret, and generate human language.
**Detailed Explanation**: NLP combines computational linguistics with statistical, machine learning, and deep learning models to process human language data. It's used for tasks like translation, sentiment analysis, and chatbots.
**Example**: Google Translate converting text from one language to another.

### Computer Vision
**Definition**: A field of AI that enables computers to derive meaningful information from digital images, videos, and other visual inputs.
**Detailed Explanation**: Computer vision uses pattern recognition and deep learning to recognize what's in a picture or video. It involves methods for acquiring, processing, analyzing, and understanding digital images.
**Example**: Facial recognition systems used for unlocking smartphones.

## Machine Learning Types

### Supervised Learning
**Definition**: A type of machine learning where the model is trained on labeled data.
**Detailed Explanation**: In supervised learning, each training example is a pair consisting of an input object (typically a vector) and a desired output value (also called the supervisory signal). The algorithm learns to map inputs to outputs.
**Example**: Predicting house prices based on features like size, location, and number of bedrooms.

### Unsupervised Learning
**Definition**: A type of machine learning that looks for patterns in unlabeled data.
**Detailed Explanation**: The system is not told the "right answer" - it must find structure in the input data on its own. Common techniques include clustering and association.
**Example**: Customer segmentation in marketing based on purchasing behavior.

### Reinforcement Learning
**Definition**: A type of machine learning where an agent learns to make decisions by taking actions in an environment to maximize some notion of cumulative reward.
**Detailed Explanation**: The agent learns from the consequences of its actions, rather than from being taught explicit examples. It's about taking suitable action to maximize reward in a particular situation.
**Example**: A computer program that learns to play chess by playing against itself.

## Development Tools

### Python
**Definition**: A high-level, interpreted programming language known for its simplicity and readability.
**Detailed Explanation**: Python is widely used in AI/ML due to its extensive libraries (like NumPy, Pandas, TensorFlow) and frameworks that simplify complex computations and algorithm implementations.

### Anaconda
**Definition**: A distribution of Python and R for scientific computing and data science.
**Detailed Explanation**: Anaconda simplifies package management and deployment. It comes with over 1,500 packages, including conda, which helps manage package versions and environments.

### Jupyter Notebook
**Definition**: An open-source web application that allows creation and sharing of documents containing live code, equations, visualizations, and narrative text.
**Detailed Explanation**: Widely used in data science for data cleaning and transformation, numerical simulation, statistical modeling, and machine learning.

### Git
**Definition**: A distributed version control system for tracking changes in source code during software development.
**Detailed Explanation**: Git enables multiple developers to work on the same codebase simultaneously without conflicts, maintaining a complete history of changes.

### GitHub
**Definition**: A web-based platform for version control and collaboration using Git.
**Detailed Explanation**: Provides hosting for software development and version control using Git, along with features like bug tracking, feature requests, task management, and wikis for projects.

## AI Ethics Concepts

### Bias in AI
**Definition**: Systematic and unfair discrimination in the operation of an AI system.
**Detailed Explanation**: Occurs when an AI system produces results that are systematically prejudiced due to erroneous assumptions in the machine learning process, often reflecting historical or social inequities present in the training data.

### Transparency
**Definition**: The principle that AI systems should be understandable by users and developers.
**Detailed Explanation**: Involves making the decision-making processes of AI systems accessible and comprehensible to stakeholders, which is crucial for building trust and accountability.

### Accountability
**Definition**: The responsibility for decisions made by AI systems.
**Detailed Explanation**: Involves ensuring there are mechanisms in place to determine who is responsible for the outcomes of AI system decisions and to address any negative impacts.

### Privacy
**Definition**: The protection of personal information and the right to control how it's collected and used.
**Detailed Explanation**: In AI, this involves ensuring that data collection, storage, and processing comply with privacy regulations and ethical standards.

## Technical Terms

### Algorithm
**Definition**: A set of rules or instructions designed to perform a specific task or solve a particular problem.
**Detailed Explanation**: In AI/ML, algorithms are the mathematical procedures that process data and learn patterns from it.

### Model
**Definition**: A mathematical representation of a real-world process.
**Detailed Explanation**: In machine learning, a model is the output of the training process, which can then be used to make predictions on new data.

### Training Data
**Definition**: The dataset used to train a machine learning model.
**Detailed Explanation**: This is the data from which the model learns the underlying patterns and relationships.

### Features
**Definition**: Individual measurable properties or characteristics of the data being analyzed.
**Detailed Explanation**: In machine learning, features are the input variables used to make predictions.

### Labels
**Definition**: The output or target variable that the model is trying to predict.
**Detailed Explanation**: In supervised learning, each training example is paired with a label that represents the desired output.

### Overfitting
**Definition**: When a model learns the training data too well, including its noise and outliers, leading to poor performance on new data.
**Detailed Explanation**: Occurs when a model is too complex relative to the amount and noisiness of the training data.

### Underfitting
**Definition**: When a model is too simple to capture the underlying structure of the data, leading to poor performance on both training and test data.
**Detailed Explanation**: Occurs when a model is too simple to capture the complexity of the data.

### Hyperparameters
**Definition**: Configuration settings used to structure the learning process of a model.
**Detailed Explanation**: These are parameters whose values are set before the learning process begins, as opposed to parameters that the model learns during training.

### Neural Network
**Definition**: A series of algorithms that attempt to recognize underlying relationships in a set of data through a process that mimics the way the human brain operates.
**Detailed Explanation**: Composed of layers of interconnected nodes (neurons) that process input data and pass information to subsequent layers.

### Backpropagation
**Definition**: An algorithm used for training neural networks.
**Detailed Explanation**: It calculates the gradient of the loss function with respect to the weights in the network, allowing for weight adjustments that minimize the loss.

### Gradient Descent
**Definition**: An optimization algorithm used to minimize some function by iteratively moving in the direction of steepest descent.
**Detailed Explanation**: In machine learning, it's used to update the parameters of the model to minimize the error or loss function.

### Loss Function
**Definition**: A method of evaluating how well a machine learning algorithm models the given data.
**Detailed Explanation**: It measures how far the model's predictions are from the actual values, providing a way to adjust the model's parameters to improve performance.
