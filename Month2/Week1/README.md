# Week 1: Linear Regression

## Learning Objectives
By the end of this week, you will be able to:
- Understand the mathematical foundation of simple and multiple linear regression
- Implement gradient descent from scratch for linear regression
- Evaluate regression models using appropriate metrics (R², MSE, RMSE)
- Apply polynomial regression and understand the bias-variance tradeoff
- Identify and handle overfitting in regression models

## Topics Covered
1. **Simple Linear Regression**
   - Hypothesis function
   - Cost function (Mean Squared Error)
   - Gradient descent algorithm

2. **Multiple Linear Regression**
   - Vectorized implementation
   - Feature scaling and normalization
   - Handling categorical variables

3. **Model Evaluation**
   - R-squared (Coefficient of Determination)
   - Mean Squared Error (MSE)
   - Root Mean Squared Error (RMSE)

4. **Polynomial Regression**
   - Feature engineering
   - Overfitting and underfitting
   - Regularization techniques (L1/L2)

## Directory Structure
```
Week1/
├── exercises/               # Coding exercises
│   ├── exercise1_simple_linear_regression.py
│   ├── exercise2_multiple_regression.py
│   ├── exercise3_gradient_descent.py
│   └── solutions/           # Exercise solutions
│       ├── exercise1_solution.py
│       ├── exercise2_solution.py
│       └── exercise3_solution.py
├── notebooks/               # Jupyter notebooks
│   ├── Linear_Regression_Implementation.ipynb
│   └── Polynomial_Regression_Demo.ipynb
├── resources/               # Additional resources
│   ├── cheatsheets/
│   │   ├── linear_regression_formulas.md
│   │   └── numpy_cheatsheet.md
│   └── sample_datasets/
│       ├── housing_prices.csv
│       └── polynomial_data.csv
├── datasets/                # Datasets for exercises
├── lecture_notes.txt        # Detailed lecture notes
└── requirements.txt         # Python dependencies
```

## Getting Started
1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Start with the lecture notes and then proceed to the exercises.

## Prerequisites
- Basic Python programming
- NumPy and Pandas fundamentals
- Basic understanding of linear algebra and statistics

## Additional Resources
- [Stanford CS229: Linear Regression](http://cs229.stanford.edu/notes-spring2019/public/cs229-notes1.pdf)
- [Google's Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course/descending-into-ml/video-lecture)
- [Scikit-learn Linear Models](https://scikit-learn.org/stable/modules/linear_model.html)

## Exercises
Complete the exercises in order:
1. Implement simple linear regression from scratch
2. Extend to multiple linear regression
3. Implement gradient descent optimization
4. Apply polynomial regression to non-linear data

## Project
Build a housing price prediction model using the provided dataset, implementing all the concepts learned this week.

## Assessment
- Complete all coding exercises (60%)
- Submit the housing price prediction project (30%)
- Peer code review (10%)
