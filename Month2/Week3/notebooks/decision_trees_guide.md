# Decision Trees and Ensemble Methods - Comprehensive Guide

## 1. Introduction

This guide provides a comprehensive look at decision trees and ensemble methods, complete with Python code examples. The content is structured to be easily converted into a Jupyter notebook.

## 2. Setup and Data Preparation

### 2.1 Import Required Libraries

```python
# Core libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn modules
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree, export_text
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    mean_squared_error, r2_score
)

# Additional libraries
import xgboost as xgb
import lightgbm as lgb
import shap

# Configuration
np.random.seed(42)
plt.style.use('seaborn-v0_8-whitegrid')
%matplotlib inline
```

### 2.2 Generate Synthetic Datasets

```python
def generate_classification_data():
    """Generate synthetic classification data."""
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_classes=2,
        class_sep=0.8,
        random_state=42
    )
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def generate_regression_data():
    """Generate synthetic regression data."""
    X, y = make_regression(
        n_samples=1000,
        n_features=10,
        n_informative=5,
        noise=10.0,
        random_state=42
    )
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Generate datasets
X_clf_train, X_clf_test, y_clf_train, y_clf_test = generate_classification_data()
X_reg_train, X_reg_test, y_reg_train, y_reg_test = generate_regression_data()
```

## 3. Decision Trees

### 3.1 Decision Tree Classifier

```python
def train_decision_tree(X_train, y_train, max_depth=3):
    """Train and return a decision tree classifier."""
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

# Train the model
dt_clf = train_decision_tree(X_clf_train, y_clf_train, max_depth=3)

def evaluate_classifier(model, X_test, y_test):
    """Evaluate and print classification metrics."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_test, y_pred), 
                annot=True, fmt='d', cmap='Blues',
                xticklabels=['Class 0', 'Class 1'],
                yticklabels=['Class 0', 'Class 1'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()

# Evaluate the model
evaluate_classifier(dt_clf, X_clf_test, y_clf_test)
```

### 3.2 Visualizing the Decision Tree

```python
def visualize_decision_tree(model, feature_names=None, class_names=None):
    """Visualize the decision tree."""
    plt.figure(figsize=(20, 10))
    plot_tree(
        model,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        fontsize=10,
        proportion=True
    )
    plt.title('Decision Tree Visualization')
    plt.show()
    
    # Print text representation
    tree_rules = export_text(
        model,
        feature_names=[f'Feature {i}' for i in range(X_clf_train.shape[1])]
    )
    print("Decision Tree Rules:")
    print(tree_rules)

# Visualize the tree
visualize_decision_tree(
    dt_clf,
    feature_names=[f'Feature {i}' for i in range(X_clf_train.shape[1])],
    class_names=['Class 0', 'Class 1']
)
```

### 3.3 Decision Tree Regressor

```python
def train_decision_tree_regressor(X_train, y_train, max_depth=3):
    """Train and return a decision tree regressor."""
    model = DecisionTreeRegressor(
        max_depth=max_depth,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

def evaluate_regressor(model, X_test, y_test):
    """Evaluate and print regression metrics."""
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R² Score: {r2:.3f}")
    
    # Plot actual vs predicted values
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], 
             [y_test.min(), y_test.max()], 
             'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted Values')
    plt.show()
    
    # Plot feature importance
    if hasattr(model, 'feature_importances_'):
        plt.figure(figsize=(10, 6))
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.title('Feature Importances')
        plt.bar(range(X_test.shape[1]), importances[indices])
        plt.xticks(range(X_test.shape[1]), 
                  [f'Feature {i}' for i in indices], 
                  rotation=90)
        plt.tight_layout()
        plt.show()

# Train and evaluate the regressor
dt_reg = train_decision_tree_regressor(X_reg_train, y_reg_train, max_depth=3)
evaluate_regressor(dt_reg, X_reg_test, y_reg_test)
```

### 3.4 Understanding Tree Depth

```python
def plot_tree_depth_impact(X_train, X_test, y_train, y_test, max_depth_range=range(1, 11)):
    """Plot the impact of tree depth on model performance."""
    train_scores = []
    test_scores = []
    
    for depth in max_depth_range:
        # Train model
        model = DecisionTreeClassifier(max_depth=depth, random_state=42)
        model.fit(X_train, y_train)
        
        # Record scores
        train_scores.append(model.score(X_train, y_train))
        test_scores.append(model.score(X_test, y_test))
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(max_depth_range, train_scores, 'b-', label='Training Accuracy')
    plt.plot(max_depth_range, test_scores, 'r-', label='Test Accuracy')
    plt.xlabel('Tree Depth')
    plt.ylabel('Accuracy')
    plt.title('Impact of Tree Depth on Model Performance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# Plot the impact of tree depth
plot_tree_depth_impact(X_clf_train, X_clf_test, y_clf_train, y_clf_test)
```

## 4. Random Forests

### 4.1 Random Forest Classifier

```python
def train_random_forest(X_train, y_train, n_estimators=100, max_depth=None):
    """Train and return a random forest classifier."""
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model

# Train the model
rf_clf = train_random_forest(X_clf_train, y_clf_train, n_estimators=100, max_depth=5)

# Evaluate the model
evaluate_classifier(rf_clf, X_clf_test, y_clf_test)
```

### 4.2 Feature Importance in Random Forest

```python
def plot_feature_importance(model, feature_names=None):
    """Plot feature importance from a tree-based model."""
    if feature_names is None:
        feature_names = [f'Feature {i}' for i in range(X_clf_train.shape[1])]
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(12, 6))
    plt.title('Feature Importances')
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), 
              [feature_names[i] for i in indices], 
              rotation=90)
    plt.tight_layout()
    plt.show()

# Plot feature importance
plot_feature_importance(rf_clf)
```

## 5. Gradient Boosting

### 5.1 XGBoost Classifier

```python
def train_xgboost(X_train, y_train, n_estimators=100, max_depth=3, learning_rate=0.1):
    """Train and return an XGBoost classifier."""
    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        objective='binary:logistic',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model

# Train the model
xgb_clf = train_xgboost(X_clf_train, y_clf_train, n_estimators=100, max_depth=3)

# Evaluate the model
evaluate_classifier(xgb_clf, X_clf_test, y_clf_test)
```

### 5.2 LightGBM Classifier

```python
def train_lightgbm(X_train, y_train, n_estimators=100, max_depth=3, learning_rate=0.1):
    """Train and return a LightGBM classifier."""
    model = lgb.LGBMClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        objective='binary',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model

# Train the model
lgbm_clf = train_lightgbm(X_clf_train, y_clf_train, n_estimators=100, max_depth=3)

# Evaluate the model
evaluate_classifier(lgbm_clf, X_clf_test, y_clf_test)
```

## 6. Model Interpretation with SHAP

### 6.1 SHAP Values for Model Interpretation

```python
def explain_model_with_shap(model, X_train, X_test):
    """Explain model predictions using SHAP values."""
    # Create explainer
    explainer = shap.TreeExplainer(model)
    
    # Calculate SHAP values
    shap_values = explainer.shap_values(X_test)
    
    # Summary plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, plot_type="bar")
    plt.title('Feature Importance (SHAP values)')
    plt.tight_layout()
    plt.show()
    
    # Dependence plot for the most important feature
    if isinstance(shap_values, list):  # For binary classification
        shap_values = shap_values[1]  # Take SHAP values for class 1
    
    # Get the index of the most important feature
    most_important_feature = np.abs(shap_values).mean(0).argmax()
    
    plt.figure(figsize=(10, 6))
    shap.dependence_plot(
        most_important_feature,
        shap_values,
        X_test,
        feature_names=[f'Feature {i}' for i in range(X_test.shape[1])]
    )
    plt.tight_layout()
    plt.show()

# Explain the XGBoost model
explain_model_with_shap(xgb_clf, X_clf_train, X_clf_test)
```

## 7. Hyperparameter Tuning

### 7.1 Grid Search for Decision Tree

```python
def tune_decision_tree(X_train, y_train):
    """Perform grid search for decision tree hyperparameters."""
    param_grid = {
        'max_depth': [3, 5, 7, 10, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'criterion': ['gini', 'entropy']
    }
    
    grid_search = GridSearchCV(
        DecisionTreeClassifier(random_state=42),
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    
    print("Best parameters:", grid_search.best_params_)
    print("Best cross-validation score: {:.3f}".format(grid_search.best_score_))
    
    return grid_search.best_estimator_

# Tune the decision tree
best_dt = tune_decision_tree(X_clf_train, y_clf_train)

# Evaluate the best model
evaluate_classifier(best_dt, X_clf_test, y_clf_test)
```

## 8. End-to-End Project: Credit Risk Modeling

### 8.1 Load and Prepare the Dataset

```python
def load_credit_data():
    """Load and prepare the credit risk dataset."""
    # For demonstration, we'll use the same synthetic data
    # In practice, you would load a real dataset here
    X, y = make_classification(
        n_samples=10000,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        weights=[0.9, 0.1],  # Imbalanced classes
        random_state=42
    )
    
    # Create feature names
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    
    # Convert to DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    return df, feature_names

# Load the data
df, feature_names = load_credit_data()

# Display basic info
print("Dataset shape:", df.shape)
print("\nClass distribution:")
print(df['target'].value_counts(normalize=True))

# Split the data
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

### 8.2 Train and Evaluate the Final Model

```python
def train_final_model(X_train, y_train):
    """Train the final model with the best parameters."""
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=sum(y_train == 0) / sum(y_train == 1),  # Handle class imbalance
        objective='binary:logistic',
        random_state=42,
        n_jobs=-1
    )
    
    # Train with early stopping
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        eval_metric='auc',
        early_stopping_rounds=20,
        verbose=50
    )
    
    return model

# Train the final model
final_model = train_final_model(X_train, y_train)

# Evaluate the final model
evaluate_classifier(final_model, X_test, y_test)

# Plot feature importance
plot_feature_importance(final_model, feature_names=feature_names)

# Explain with SHAP
explain_model_with_shap(final_model, X_train, X_test)
```

## 9. Conclusion

This guide has covered the essential aspects of decision trees and ensemble methods, including:

1. **Decision Trees**: Building blocks for more complex models
2. **Random Forests**: Ensemble of decision trees with bagging
3. **Gradient Boosting**: Sequential building of trees to correct errors
4. **Model Interpretation**: Understanding model decisions with SHAP values
5. **Hyperparameter Tuning**: Optimizing model performance
6. **End-to-End Project**: Applying concepts to a real-world problem

To convert this guide into a Jupyter notebook, simply copy each code block into a separate code cell in the notebook. The markdown sections provide explanations and context for each code block.

## 10. Additional Resources

- [Scikit-learn Documentation - Decision Trees](https://scikit-learn.org/stable/modules/tree.html)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [SHAP Documentation](https://shap.readthedocs.io/)
- [Interpretable Machine Learning Book](https://christophm.github.io/interpretable-ml-book/)
- [Elements of Statistical Learning](https://web.stanford.edu/~hastie/ElemStatLearn/)
