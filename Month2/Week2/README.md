# Week 2: Logistic Regression and K-Nearest Neighbors (KNN)

## 🎯 Weekly Objectives

By the end of this week, students will be able to:
- Understand and implement Logistic Regression for binary classification
- Apply K-Nearest Neighbors (KNN) for both classification and regression
- Evaluate classification models using appropriate metrics
- Handle class imbalance in classification problems
- Interpret model coefficients and feature importance

## 📚 Prerequisites

- Linear Regression concepts (from Week 1)
- Python programming with NumPy and Pandas
- Basic understanding of probability and statistics
- Familiarity with Jupyter Notebooks

## 🏆 Student Expectations

### Active Participation
- Complete all pre-class readings and exercises
- Engage in hands-on coding activities
- Participate in group discussions and peer code reviews
- Ask questions and seek clarification

### Time Commitment
- 4-6 hours of lecture and lab time
- 6-8 hours of self-study and practice
- 4-6 hours for the weekly project

### Deliverables
- Completed coding exercises (due Wednesday)
- Weekly quiz (due Friday)
- Classification project (due Sunday)
- Reflection journal entry (due Sunday)

## 📅 Weekly Schedule

### Day 1: Introduction to Classification & Logistic Regression
- From regression to classification
- The sigmoid function and decision boundaries
- Cost function for logistic regression
- Gradient descent for logistic regression

### Day 2: Evaluating Classification Models
- Confusion matrix and metrics (accuracy, precision, recall, F1)
- ROC curves and AUC-ROC
- Precision-Recall curves
- Handling class imbalance

### Day 3: K-Nearest Neighbors (KNN)
- Intuition behind instance-based learning
- Distance metrics (Euclidean, Manhattan, etc.)
- Choosing the right k value
- KNN for regression

### Day 4: Model Comparison and Practical Considerations
- Comparing Logistic Regression and KNN
- Feature scaling and preprocessing
- Cross-validation for model selection
- Real-world applications

### Day 5: Project Work and Review
- Project implementation
- Code reviews
- Q&A session
- Week in review

## 📝 Assessments

1. **Weekly Quiz (20%)**
   - Covers theoretical concepts
   - Multiple-choice and short-answer questions
   - Time-limited (60 minutes)

2. **Coding Exercises (30%)**
   - Implementing logistic regression from scratch
   - Applying KNN to real datasets
   - Model evaluation and interpretation

3. **Classification Project (40%)**
   - End-to-end classification project
   - Real-world dataset provided
   - Report submission with analysis and visualizations

4. **Participation (10%)**
   - Class attendance
   - Forum participation
   - Peer feedback

## 📚 Reading Recommendations

### Required Reading
1. **"An Introduction to Statistical Learning" (ISL) by James, Witten, Hastie, and Tibshirani**
   - *Chapter 4: Classification* - This foundational chapter provides the mathematical bedrock for understanding logistic regression, linear discriminant analysis, and the broader landscape of classification problems. Pay special attention to the intuitive explanations of the logistic function and how it elegantly maps linear combinations of features to probability estimates between 0 and 1. The discussion on the Bayes classifier and decision boundaries will give you a theoretical framework that will serve you throughout your machine learning journey.
   
   - *Chapter 2: Statistical Learning Review* - While you've encountered this material before, revisiting the model assessment section with your new perspective on classification will reveal deeper insights. Focus particularly on the bias-variance tradeoff as it applies to classification problems, and how different algorithms manage this fundamental tension in their own unique ways.

2. **"Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow" by Aurélien Géron**
   - *Chapter 3: Classification* - Géron's practical approach brings classification algorithms to life with clear examples and scikit-learn implementations. The section on performance metrics is particularly valuable, as it will help you understand why accuracy alone can be misleading and how to choose the right metric for different business contexts. The discussion on multiclass and multilabel classification will expand your understanding beyond binary classification.
   
   - *Chapter 2: End-to-End Project* - Revisit this chapter with classification in mind. The structured approach to problem-solving presented here is universally applicable, but now you'll notice how considerations like class imbalance and decision thresholds become crucial in classification tasks compared to regression problems.

### Additional Resources
1. **"Python Data Science Handbook" by Jake VanderPlas**
   - *Chapter 5: Machine Learning* - VanderPlas has a gift for making complex concepts accessible. His treatment of K-Nearest Neighbors is particularly illuminating, showing both the simplicity and power of this versatile algorithm. The visualizations of decision boundaries for different values of k will help you develop an intuitive understanding of the bias-variance tradeoff in action.

2. **Research Papers (For the Curious Mind)**
   - *"A Few Useful Things to Know About Machine Learning"* by Pedro Domingos - This highly cited paper distills practical wisdom that's especially relevant when working with classification algorithms. The discussion of the "no free lunch" theorem will help you understand why logistic regression might outperform more complex models in certain scenarios.
   
   - *"An Empirical Study of the Naive Bayes Classifier"* by Irina Rish - While focused on Naive Bayes, this paper provides valuable insights into the behavior of simple probabilistic classifiers, which will deepen your understanding of logistic regression's strengths and limitations.

3. **Interactive Learning**
   - *Distill.pub Articles* - The interactive visualizations in articles like "A Visual Introduction to Machine Learning" can help solidify your understanding of how logistic regression and KNN make decisions, especially when dealing with non-linear boundaries and different feature spaces.
   
   - *Google's Machine Learning Crash Course* - The classification sections offer interactive exercises that let you experiment with different classification algorithms and immediately see the impact of various parameters and preprocessing steps.
   - Excellent practical examples using scikit-learn

2. **"Pattern Recognition and Machine Learning" by Christopher Bishop**
   - Chapter 4: Linear Models for Classification
   - More mathematical treatment of logistic regression

3. **Research Papers**
   - "A Few Useful Things to Know About Machine Learning" by Pedro Domingos
   - "On the Dangers of Cross-Validation. An Experimental Evaluation" by Gavin Cawley

### Online Courses & Tutorials
1. **Coursera: Machine Learning by Andrew Ng**
   - Week 3: Classification and Representation
   - Week 4: Neural Networks: Representation

2. **Fast.ai: Practical Deep Learning for Coders**
   - Lesson 1: Your First Model
   - Excellent for practical implementation

## 🔍 Datasets for Practice
1. **Iris Dataset** - Basic classification
2. **Titanic Dataset** - Binary classification with real-world data
3. **MNIST** - Handwritten digit recognition (for KNN)
4. **Breast Cancer Wisconsin** - Medical classification task

## 🛠 Tools and Libraries
- scikit-learn: `LogisticRegression`, `KNeighborsClassifier`
- imbalanced-learn: For handling class imbalance
- seaborn: For visualization of decision boundaries
- Yellowbrick: For model visualization and evaluation

## 📝 Weekly Reflection
At the end of the week, reflect on:
- The differences between regression and classification
- When to use logistic regression vs. KNN
- The importance of proper evaluation metrics
- Challenges faced and how you overcame them
