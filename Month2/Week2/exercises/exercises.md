# Week 2 Exercises: Logistic Regression and KNN

## Exercise 1: Implementing Logistic Regression from Scratch

### 1.1 Sigmoid Function
Implement the sigmoid function that maps any real-valued number to a value between 0 and 1.

```python
def sigmoid(z):
    """
    Compute the sigmoid of z
    
    Parameters:
    z -- A scalar or numpy array of any size
    
    Returns:
    s -- sigmoid(z)
    """
    # Your code here
    pass
```

### 1.2 Cost Function and Gradient
Implement the cost function and gradient for logistic regression.

```python
def compute_cost(X, y, w, b):
    """
    Compute the cost function for logistic regression
    
    Parameters:
    X -- Input data of shape (m, n)
    y -- True labels of shape (m, 1)
    w -- Weights of shape (n, 1)
    b -- Bias (a scalar)
    
    Returns:
    cost -- The computed cost
    """
    m = X.shape[0]
    # Your code here
    return cost

def compute_gradient(X, y, w, b):
    """
    Compute the gradient for logistic regression
    
    Parameters:
    X -- Input data of shape (m, n)
    y -- True labels of shape (m, 1)
    w -- Weights of shape (n, 1)
    b -- Bias (a scalar)
    
    Returns:
    dj_dw -- Gradient of the cost w.r.t. w
    dj_db -- Gradient of the cost w.r.t. b
    """
    m = X.shape[0]
    # Your code here
    return dj_dw, dj_db
```

## Exercise 2: KNN Implementation

### 2.1 Distance Metrics
Implement the following distance metrics:

```python
def euclidean_distance(a, b):
    """
    Compute the Euclidean distance between two points
    
    Parameters:
    a, b -- numpy arrays of the same shape
    
    Returns:
    distance -- The Euclidean distance between a and b
    """
    # Your code here
    pass

def manhattan_distance(a, b):
    """
    Compute the Manhattan distance between two points
    """
    # Your code here
    pass
```

### 2.2 KNN Classifier
Implement a simple KNN classifier:

```python
class KNN:
    def __init__(self, k=5, distance_metric='euclidean'):
        """
        Initialize the KNN classifier
        
        Parameters:
        k -- number of neighbors to consider
        distance_metric -- 'euclidean' or 'manhattan'
        """
        self.k = k
        self.distance_metric = distance_metric
        
    def fit(self, X, y):
        """
        Store the training data
        """
        self.X_train = X
        self.y_train = y
        
    def predict(self, X):
        """
        Predict the class labels for the input data
        """
        # Your code here
        pass
```

## Exercise 3: Model Evaluation

### 3.1 Confusion Matrix
Implement a function to compute the confusion matrix:

```python
def confusion_matrix(y_true, y_pred, labels=None):
    """
    Compute the confusion matrix
    
    Parameters:
    y_true -- Ground truth labels
    y_pred -- Predicted labels
    labels -- List of labels to index the matrix
    
    Returns:
    cm -- Confusion matrix as a 2D numpy array
    """
    # Your code here
    pass
```

### 3.2 Classification Report
Implement a function to generate a classification report:

```python
def classification_report(y_true, y_pred, target_names=None):
    """
    Generate a classification report with precision, recall, f1-score
    """
    # Your code here
    pass
```

## Exercise 4: Real-world Application

### 4.1 Breast Cancer Classification
Using the Breast Cancer Wisconsin dataset, implement a logistic regression model and a KNN classifier to predict whether a tumor is malignant or benign.

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Your code here:
# 1. Split the data into training and testing sets
# 2. Scale the features
# 3. Train both models
# 4. Evaluate using appropriate metrics
# 5. Compare the performance
```

## Exercise 5: Advanced Topics

### 5.1 Handling Class Imbalance
Implement a solution to handle class imbalance using:
1. Random oversampling
2. SMOTE (Synthetic Minority Over-sampling Technique)
3. Class weights in logistic regression

### 5.2 Hyperparameter Tuning
Perform hyperparameter tuning for both models using GridSearchCV and RandomizedSearchCV. Compare the results.

## Submission Guidelines
1. Complete all exercises in a Jupyter notebook
2. Include markdown cells explaining your approach
3. Visualize the results where appropriate
4. Submit your notebook and a PDF export by the deadline
