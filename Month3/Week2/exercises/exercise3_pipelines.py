"""
Exercise 3: Building ML Pipelines with Feature Selection

This exercise demonstrates how to build ML pipelines with feature selection
and dimensionality reduction steps.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

def load_data():
    """Load and split the breast cancer dataset."""
    data = load_breast_cancer()
    X, y = data.data, data.target
    return train_test_split(X, y, test_size=0.2, random_state=42)

def build_pipeline():
    """Build a pipeline with feature selection and classification."""
    # Create a feature union for feature selection and PCA
    features = FeatureUnion([
        ('pca', PCA()),
        ('select_best', SelectKBest(score_func=f_classif))
    ])
    
    # Create the pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('features', features),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    
    return pipeline

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and print metrics."""
    y_pred = model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

def main():
    # Load and split data
    X_train, X_test, y_train, y_test = load_data()
    
    # Build and train the pipeline
    print("Building and training the pipeline...")
    pipeline = build_pipeline()
    
    # Define parameter grid for grid search
    param_grid = {
        'features__pca__n_components': [5, 10, 15],
        'features__select_best__k': [5, 10, 15],
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [None, 10, 20]
    }
    
    # Perform grid search
    print("Performing grid search...")
    grid_search = GridSearchCV(
        pipeline, 
        param_grid, 
        cv=5, 
        n_jobs=-1, 
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    # Print best parameters
    print("\nBest parameters:")
    print(grid_search.best_params_)
    
    # Evaluate the best model
    print("\nBest model evaluation:")
    evaluate_model(grid_search.best_estimator_, X_test, y_test)
    
    # Compare with a simple model
    print("\nSimple model (no feature selection):")
    simple_model = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    simple_model.fit(X_train, y_train)
    evaluate_model(simple_model, X_test, y_test)

if __name__ == "__main__":
    main()
