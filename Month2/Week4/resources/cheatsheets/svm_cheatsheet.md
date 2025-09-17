# SVM and Model Evaluation Cheatsheet

## Support Vector Machines (SVM)

### Basic Usage
```python
from sklearn.svm import SVC, SVR

# Classification
clf = SVC(kernel='rbf', C=1.0, gamma='scale')
clf.fit(X_train, y_train)

# Regression
reg = SVR(kernel='rbf', C=1.0, gamma='scale')
reg.fit(X_train, y_train)
```

### Common Parameters
- `C`: Regularization parameter (smaller values = stronger regularization)
- `kernel`: 'linear', 'poly', 'rbf', 'sigmoid'
- `gamma`: Kernel coefficient for 'rbf', 'poly', 'sigmoid'
- `degree`: Degree of polynomial kernel
- `class_weight`: For imbalanced classes

## Model Evaluation

### Classification Metrics
```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve
)

# Basic metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Classification report
print(classification_report(y_true, y_pred))
```

### Regression Metrics
```python
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score
)

mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)
```

### Cross-Validation
```python
from sklearn.model_selection import (
    cross_val_score, cross_validate,
    GridSearchCV, RandomizedSearchCV
)

# Basic cross-validation
scores = cross_val_score(estimator, X, y, cv=5)

# Grid search
param_grid = {'C': [0.1, 1, 10], 'gamma': [1, 0.1, 0.01]}
grid = GridSearchCV(SVC(), param_grid, cv=5)
grid.fit(X_train, y_train)
best_params = grid.best_params_
```

### Handling Class Imbalance
```python
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

# Using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# In pipeline
pipeline = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('classifier', SVC())
])
```

### Visualization
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Confusion matrix heatmap
sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Actual')

# ROC Curve
fpr, tpr, _ = roc_curve(y_true, y_scores)
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
```
