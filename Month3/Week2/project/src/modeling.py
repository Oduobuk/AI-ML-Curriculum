"""
Modeling utilities for the Fashion MNIST project.
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV

def train_random_forest(X_train, y_train, **kwargs):
    """
    Train a Random Forest classifier.
    
    Args:
        X_train: Training features
        y_train: Training labels
        **kwargs: Additional arguments for RandomForestClassifier
        
    Returns:
        RandomForestClassifier: Trained model
    """
    model = RandomForestClassifier(random_state=42, **kwargs)
    model.fit(X_train, y_train)
    return model

def train_svm(X_train, y_train, **kwargs):
    """
    Train an SVM classifier.
    
    Args:
        X_train: Training features
        y_train: Training labels
        **kwargs: Additional arguments for SVC
        
    Returns:
        SVC: Trained model
    """
    model = SVC(random_state=42, **kwargs)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate a model on test data.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        
    Returns:
        dict: Dictionary with evaluation metrics
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    return {
        'accuracy': accuracy,
        'report': report
    }

def tune_hyperparameters(model, param_grid, X_train, y_train, cv=5):
    """
    Perform hyperparameter tuning using grid search.
    
    Args:
        model: Model to tune
        param_grid: Parameter grid for GridSearchCV
        X_train: Training features
        y_train: Training labels
        cv: Number of cross-validation folds
        
    Returns:
        dict: Best parameters and best score
    """
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    return {
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'best_estimator': grid_search.best_estimator_
    }
