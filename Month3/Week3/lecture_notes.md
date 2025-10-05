# Week 3: Anomaly Detection & Recommendation Systems

## 1. Anomaly Detection

### 1.1 Introduction
- What are anomalies?
- Types of anomalies (point, contextual, collective)
- Applications: fraud detection, system health monitoring, etc.

### 1.2 Statistical Methods
- Z-score method
- Interquartile Range (IQR)
- Modified Z-score
- Moving averages for time series

### 1.3 Machine Learning Approaches
- **Isolation Forest**
  - Isolation principle
  - Path length as anomaly score
  - Implementation in scikit-learn

- **One-Class SVM**
  - Concept of support vectors
  - Kernel trick for non-linear boundaries
  - Nu parameter for controlling outliers

- **DBSCAN**
  - Density-based clustering
  - Core points and border points
  - Identifying noise as anomalies

### 1.4 Deep Learning Approaches
- Autoencoders for anomaly detection
- LSTM networks for sequential data
- Variational Autoencoders (VAEs)

## 2. Recommendation Systems

### 2.1 Introduction
- What are recommendation systems?
- Types: content-based, collaborative filtering, hybrid
- Evaluation metrics: RMSE, MAE, precision@k, recall@k

### 2.2 Content-Based Filtering
- Item representation using TF-IDF
- Cosine similarity
- Pros and cons
- Implementation example

### 2.3 Collaborative Filtering
- User-based vs item-based
- Matrix factorization
- Singular Value Decomposition (SVD)
- Alternating Least Squares (ALS)

### 2.4 Advanced Techniques
- Neural Collaborative Filtering
- Factorization Machines
- Session-based recommendations
- Context-aware recommendations

## 3. Practical Considerations

### 3.1 Cold Start Problem
- New user problem
- New item problem
- Solutions: hybrid approaches, content-based initialization

### 3.2 Scalability
- Handling large datasets
- Incremental learning
- Distributed computing with Spark

### 3.3 Evaluation
- Offline evaluation
- A/B testing
- Business metrics

## 4. Case Studies
1. Netflix Recommendation System
2. Amazon Product Recommendations
3. Spotify's Discover Weekly

## 5. Hands-on Exercises
1. Implement anomaly detection on credit card transactions
2. Build a movie recommendation system
3. Evaluate different recommendation algorithms

## 6. Resources
- Books:
  - "Programming Collective Intelligence" by Toby Segaran
  - "Recommender Systems: The Textbook" by Charu C. Aggarwal
- Papers:
  - "Matrix Factorization Techniques for Recommender Systems" (Netflix Prize)
  - "Variational Autoencoders for Collaborative Filtering"
- Online Courses:
  - Coursera: "Machine Learning" by Andrew Ng (recommender systems)
  - Fast.ai: Practical Deep Learning for Coders

## 7. Next Steps
- Experiment with real-world datasets
- Explore hybrid recommendation systems
- Learn about reinforcement learning for recommendations
- Stay updated with latest research (e.g., graph neural networks)
