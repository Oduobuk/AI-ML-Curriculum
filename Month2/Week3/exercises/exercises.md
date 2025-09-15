# Week 3: Decision Trees and Ensemble Methods - Exercises

## Exercise 1: Decision Tree Basics

### 1.1 Implementing a Decision Tree from Scratch

**Objective**: Understand how decision trees make decisions by implementing a simplified version from scratch.

**Tasks**:
1. Implement the Gini impurity calculation
2. Create a function to find the best split for a dataset
3. Build a recursive function to grow the tree
4. Implement prediction for a single sample

**Starter Code**:
```python
import numpy as np
from collections import Counter

class DecisionNode:
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, value=None):
        self.feature_idx = feature_idx  # Index of feature to split on
        self.threshold = threshold      # Threshold value for the split
        self.left = left                # Left subtree
        self.right = right              # Right subtree
        self.value = value              # Value if leaf node (for prediction)

def gini_impurity(y):
    """
    Calculate Gini impurity for a set of labels.
    
    Args:
        y: Array of labels
        
    Returns:
        Gini impurity (float)
    """
    # Your implementation here
    pass

def find_best_split(X, y):
    """
    Find the best split for the data.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        y: Target vector (n_samples,)
        
    Returns:
        Tuple of (best_feature_idx, best_threshold, best_gain)
    """
    # Your implementation here
    pass

class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None
    
    def fit(self, X, y, depth=0):
        """
        Build the decision tree.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (n_samples,)
            depth: Current depth in the tree
        """
        # Your implementation here
        pass
    
    def predict(self, X):
        """
        Make predictions for input samples.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            Predicted labels (n_samples,)
        """
        # Your implementation here
        pass
```

### 1.2 Decision Tree Hyperparameters

**Objective**: Understand how different hyperparameters affect decision tree performance.

**Tasks**:
1. Load the Iris dataset
2. Train decision trees with different max_depth values (1-10)
3. Plot training and validation accuracy vs. max_depth
4. Repeat for min_samples_split and min_samples_leaf
5. Analyze the trade-offs between bias and variance

**Questions**:
1. How does increasing max_depth affect model complexity?
2. What happens to the training error as max_depth increases?
3. How does min_samples_leaf help prevent overfitting?

## Exercise 2: Random Forests

### 2.1 Implementing a Random Forest from Scratch

**Objective**: Understand how random forests combine multiple decision trees.

**Tasks**:
1. Implement bootstrap sampling
2. Create a random subset of features for each tree
3. Build multiple decision trees on different samples
4. Implement majority voting for classification

**Starter Code**:
```python
import numpy as np
from sklearn.utils import resample
from sklearn.tree import DecisionTreeClassifier

class RandomForest:
    def __init__(self, n_estimators=100, max_features='sqrt', **tree_params):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.tree_params = tree_params
        self.trees = []
        self.feature_subsets = []
    
    def _bootstrap_sample(self, X, y):
        """Create a bootstrap sample of the data."""
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, size=n_samples, replace=True)
        return X[idxs], y[idxs]
    
    def _get_feature_subset(self, n_features):
        """Get a random subset of features."""
        if isinstance(self.max_features, int):
            n = min(self.max_features, n_features)
        elif self.max_features == 'sqrt':
            n = int(np.sqrt(n_features))
        else:  # 'log2' or float
            n = int(np.log2(n_features)) if self.max_features == 'log2' else int(self.max_features * n_features)
        
        return np.random.choice(n_features, size=n, replace=False)
    
    def fit(self, X, y):
        """Train the random forest."""
        n_samples, n_features = X.shape
        self.trees = []
        self.feature_subsets = []
        
        for _ in range(self.n_estimators):
            # Your implementation here
            pass
    
    def predict_proba(self, X):
        """Predict class probabilities."""
        # Your implementation here
        pass
    
    def predict(self, X):
        """Predict class labels."""
        # Your implementation here
        pass
```

### 2.2 Feature Importance Analysis

**Objective**: Understand feature importance in random forests.

**Tasks**:
1. Train a random forest on a dataset
2. Calculate and plot feature importances
3. Compare with permutation importance
4. Analyze the top important features

**Questions**:
1. How does random forest calculate feature importance?
2. What are the advantages of permutation importance?
3. How would you handle correlated features in importance analysis?

## Exercise 3: Gradient Boosting

### 3.1 Implementing Gradient Boosting from Scratch

**Objective**: Understand how gradient boosting builds an ensemble of weak learners.

**Tasks**:
1. Implement the gradient boosting algorithm for regression
2. Use decision stumps (depth-1 trees) as weak learners
3. Implement gradient calculation with different loss functions
4. Add learning rate and subsampling

**Starter Code**:
```python
import numpy as np
from sklearn.tree import DecisionTreeRegressor

class GradientBoostingRegressor:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, subsample=1.0):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.trees = []
        self.initial_prediction = None
    
    def _gradient(self, y_true, y_pred):
        """Compute the negative gradient of the loss function."""
        # For MSE loss, gradient = y_pred - y_true
        return y_pred - y_true
    
    def fit(self, X, y):
        """Train the gradient boosting model."""
        # Initialize with mean
        self.initial_prediction = np.mean(y)
        y_pred = np.full_like(y, self.initial_prediction, dtype=np.float64)
        
        for _ in range(self.n_estimators):
            # Your implementation here
            pass
    
    def predict(self, X):
        """Make predictions."""
        # Your implementation here
        pass
```

### 3.2 XGBoost Tuning

**Objective**: Learn to tune XGBoost hyperparameters effectively.

**Tasks**:
1. Load a dataset and prepare it for modeling
2. Perform grid search for key hyperparameters:
   - learning_rate
   - max_depth
   - n_estimators
   - subsample
   - colsample_bytree
3. Use early stopping to prevent overfitting
4. Visualize the effect of different parameters

**Questions**:
1. How does the learning rate affect model training?
2. What's the trade-off between n_estimators and learning_rate?
3. How does subsample help prevent overfitting?

## Exercise 4: Model Interpretation

### 4.1 SHAP Values

**Objective**: Learn to interpret complex models using SHAP values.

**Tasks**:
1. Train a random forest or gradient boosting model
2. Calculate SHAP values for the test set
3. Create summary plots and dependence plots
4. Analyze feature interactions

**Code Example**:
```python
import shap
import xgboost as xgb

# Train XGBoost model
model = xgb.XGBClassifier().fit(X_train, y_train)

# Create explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Summary plot
shap.summary_plot(shap_values, X_test)

# Dependence plot for a specific feature
shap.dependence_plot("feature_name", shap_values, X_test)
```

### 4.2 Partial Dependence Plots

**Objective**: Understand the relationship between features and predictions.

**Tasks**:
1. Create partial dependence plots for important features
2. Analyze interaction effects
3. Compare with ICE (Individual Conditional Expectation) plots

## Exercise 5: End-to-End Project

### Credit Risk Modeling

**Objective**: Apply all concepts to a real-world problem.

**Tasks**:
1. Load and explore the German Credit dataset
2. Perform feature engineering and preprocessing
3. Train and evaluate multiple models:
   - Decision Tree
   - Random Forest
   - XGBoost
4. Tune hyperparameters using cross-validation
5. Analyze feature importance and model interpretability
6. Create a simple web service for predictions

**Deliverables**:
1. Jupyter notebook with complete analysis
2. Model evaluation metrics (AUC-ROC, precision, recall, F1)
3. Feature importance analysis
4. Deployment code (Flask/FastAPI)

## Additional Challenges

1. **Handling Imbalanced Data**:
   - Implement SMOTE and other oversampling techniques
   - Use class weights in tree-based models
   - Evaluate using precision-recall curves and F1 score

2. **Time Series Forecasting**:
   - Adapt tree-based models for time series data
   - Create lag features and rolling statistics
   - Implement walk-forward validation

3. **Deployment**:
   - Create a REST API for model serving
   - Implement model versioning
   - Set up monitoring for model performance

## Submission Guidelines

1. Complete all exercises in a Jupyter notebook
2. Include detailed explanations and visualizations
3. Document your thought process and any challenges faced
4. Submit your code and a PDF export of the notebook

## Resources

- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [SHAP Documentation](https://shap.readthedocs.io/)
- [Interpretable Machine Learning](https://christophm.github.io/interpretable-ml-book/)
- [Elements of Statistical Learning](https://web.stanford.edu/~hastie/ElemStatLearn/)
