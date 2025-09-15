# Week 3: Decision Trees and Ensemble Methods

This week, we'll explore decision trees and powerful ensemble methods that build upon them, including Random Forests and Gradient Boosting. These techniques form the foundation of many winning solutions in machine learning competitions and real-world applications.

## Learning Objectives

By the end of this week, you should be able to:
- Understand the theory and intuition behind decision trees
- Implement and tune decision trees for classification and regression
- Explain and apply ensemble methods: Bagging, Random Forest, and Boosting
- Use feature importance to interpret tree-based models
- Handle imbalanced datasets with tree-based methods
- Deploy tree-based models in production

## Prerequisites
- Week 1: Python for Machine Learning
- Week 2: Logistic Regression and K-Nearest Neighbors
- Basic understanding of probability and statistics
- Familiarity with scikit-learn

## Topics Covered

### 1. Decision Trees
- Tree structure and terminology
- Splitting criteria (Gini, Entropy, MSE)
- Pruning and regularization
- Handling categorical and missing data
- Advantages and limitations

### 2. Random Forests
- Bootstrap aggregating (Bagging)
- Feature importance and selection
- Out-of-bag error estimation
- Extremely Randomized Trees (Extra-Trees)

### 3. Gradient Boosting
- Adaptive Boosting (AdaBoost)
- Gradient Boosting Machines (GBM)
- XGBoost, LightGBM, and CatBoost
- Hyperparameter tuning

### 4. Advanced Topics
- Handling imbalanced data
- Feature engineering for tree-based models
- Model interpretation and explainability
- Deployment considerations

## Getting Started

### Setup Environment
1. Create a new conda environment:
   ```bash
   conda create -n ml_week3 python=3.9
   conda activate ml_week3
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Required Datasets
Download the following datasets to the `datasets/` directory:
1. [Titanic Dataset](https://www.kaggle.com/c/titanic/data) - For classification tasks
2. [House Prices](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) - For regression tasks
3. [Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud) - For imbalanced classification

## Weekly Schedule

### Day 1: Decision Trees Fundamentals
- Morning: Theory and Mathematics
- Afternoon: Implementing Decision Trees in scikit-learn
- Exercise: Build a decision tree classifier from scratch

### Day 2: Random Forests
- Morning: Ensemble Methods and Bagging
- Afternoon: Implementing Random Forests
- Exercise: Feature importance analysis

### Day 3: Gradient Boosting
- Morning: Boosting Algorithms
- Afternoon: XGBoost and LightGBM
- Exercise: Hyperparameter tuning with Optuna

### Day 4: Advanced Topics
- Morning: Handling Imbalanced Data
- Afternoon: Model Interpretation
- Exercise: SHAP values for model explainability

### Day 5: Project Work
- End-to-end project implementation
- Code review and best practices
- Deployment considerations

## Resources

### Required Reading
1. [Scikit-learn Decision Trees Documentation](https://scikit-learn.org/stable/modules/tree.html)
2. [Random Forest Paper](https://link.springer.com/article/10.1023/A:1010933404324) by Leo Breiman
3. [XGBoost Paper](https://arxiv.org/abs/1603.02754)

### Recommended Reading
1. [The Elements of Statistical Learning](https://web.stanford.edu/~hastie/ElemStatLearn/), Chapter 9 (Tree-Based Methods) and 10 (Boosting and Additive Trees)
2. [Interpretable Machine Learning](https://christophm.github.io/interpretable-ml-book/), Chapter 5 (Decision Trees) and 6 (Feature Importance)

### Additional Resources
- [Awesome Decision Tree Papers](https://github.com/benedekrozemberczki/awesome-decision-tree-papers)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)

## Assessment
- Weekly quiz (20%)
- Programming assignments (40%)
- End-of-week project (40%)

## Getting Help
- Use the course discussion forum for questions
- Attend office hours (TBA)
- Form study groups with your peers

## License
This curriculum is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
