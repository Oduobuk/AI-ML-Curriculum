"""
<<<<<<< HEAD
Exercise 3: Dimensionality Reduction Pipelines

In this exercise, you'll build end-to-end ML pipelines that incorporate
dimensionality reduction and evaluate their performance.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import fetch_openml, fetch_olivetti_faces
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neighbors import KNeighborsClassifier

# Set random seed for reproducibility
np.random.seed(42)

def load_dataset(dataset_name='digits'):
    """Load and preprocess the specified dataset."""
    print(f"\n=== Loading {dataset_name} dataset ===")
    
    if dataset_name == 'digits':
        data = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
        X = data.data
        y = data.target.astype(int)
        # Take a subset for faster computation
        X, _, y, _ = train_test_split(X, y, train_size=2000, stratify=y, random_state=42)
        
    elif dataset_name == 'olivetti':
        data = fetch_olivetti_faces(shuffle=True, random_state=42)
        X = data.images.reshape((len(data.images), -1))  # Flatten images
        y = data.target
        
    elif dataset_name == 'wine':
        data = fetch_openml('wine', version=1, as_frame=False, parser='auto')
        X = data.data
        y = data.target
        
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Convert to numpy arrays if they're not already
    X = np.array(X)
    y = np.array(y)
    
    # Print dataset information
    print(f"Dataset shape: {X.shape}")
    print(f"Number of classes: {len(np.unique(y))}")
    print(f"Feature names: {data.feature_names if hasattr(data, 'feature_names') else 'N/A'}")
    
    return X, y, data

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name=""):
    """Evaluate a model and return performance metrics."""
    start_time = time.time()
    
    # Train the model
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    # Make predictions
    start_time = time.time()
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
    predict_time = time.time() - start_time
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    # Classification report
    print(f"\n=== {model_name} ===")
    print(f"Training time: {train_time:.3f}s")
    print(f"Prediction time: {predict_time:.3f}s")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    # ROC and PR curves for binary classification
    if len(np.unique(y_test)) == 2 and y_pred_proba is not None:
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        
        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba[:, 1])
        avg_precision = average_precision_score(y_test, y_pred_proba[:, 1])
        
        # Plot ROC and PR curves
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # ROC Curve
        ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title(f'ROC Curve - {model_name}')
        ax1.legend(loc='lower right')
        
        # Precision-Recall Curve
        ax2.plot(recall, precision, color='blue', lw=2, 
                label=f'PR curve (AP = {avg_precision:.2f})')
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title(f'Precision-Recall Curve - {model_name}')
        ax2.legend(loc='lower left')
        
        plt.tight_layout()
        plt.show()
    
    return {
        'model': model_name,
        'accuracy': accuracy,
        'train_time': train_time,
        'predict_time': predict_time,
    }

def build_pipeline(reducer_name='pca', n_components=2, classifier='rf'):
    """Build a pipeline with the specified dimensionality reduction and classifier."""
    # Define the reducer
    if reducer_name == 'pca':
        reducer = PCA(n_components=n_components, random_state=42)
    elif reducer_name == 'tsne':
        # Note: t-SNE doesn't have a transform method for new data
        # So we'll use it only for visualization, not in the pipeline
        reducer = None
    elif reducer_name == 'umap':
        reducer = umap.UMAP(n_components=n_components, random_state=42)
    else:
        raise ValueError(f"Unknown reducer: {reducer_name}")
    
    # Define the classifier
    if classifier == 'rf':
        clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    elif classifier == 'svm':
        clf = SVC(probability=True, random_state=42)
    elif classifier == 'lr':
        clf = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
    else:
        raise ValueError(f"Unknown classifier: {classifier}")
    
    # Build the pipeline
    if reducer:
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('reducer', reducer),
            ('classifier', clf)
        ])
    else:
        # For t-SNE, we'll handle it separately since it doesn't support transform
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', clf)
        ])
    
    return pipeline, reducer_name, classifier

def compare_reducers(X, y, reducers=['pca', 'umap'], n_components=2, classifier='rf'):
    """Compare different dimensionality reduction techniques."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    results = []
    
    for reducer_name in reducers:
        print(f"\n=== Evaluating {reducer_name.upper()} with {classifier.upper()} ===")
        
        if reducer_name == 'tsne':
            # Special handling for t-SNE since it doesn't support transform
            # We'll apply it to the training data and train a model on the reduced space
            # Then we'll evaluate on the test set without t-SNE (which is not ideal but necessary)
            
            # Reduce training data with t-SNE
            start_time = time.time()
            tsne = TSNE(n_components=n_components, random_state=42, n_jobs=-1)
            X_train_tsne = tsne.fit_transform(X_train)
            train_time = time.time() - start_time
            
            # Train a model on the reduced training data
            if classifier == 'rf':
                clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            elif classifier == 'svm':
                clf = SVC(probability=True, random_state=42)
            elif classifier == 'lr':
                clf = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
            
            clf.fit(X_train_tsne, y_train)
            
            # For test data, we can't apply t-SNE transform, so we'll use the original features
            # This is a limitation of t-SNE for production use
            y_pred = clf.predict(X_test)
            y_pred_proba = clf.predict_proba(X_test) if hasattr(clf, 'predict_proba') else None
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            
            print(f"Training time: {train_time:.3f}s")
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred))
            
            results.append({
                'reducer': 'tsne',
                'classifier': classifier,
                'accuracy': accuracy,
                'train_time': train_time,
                'predict_time': 0,  # Not measured
            })
            
        else:
            # For PCA and UMAP, we can use them in a pipeline
            pipeline, _, _ = build_pipeline(reducer_name, n_components, classifier)
            
            # Evaluate the pipeline
            metrics = evaluate_model(
                pipeline, X_train, X_test, y_train, y_test,
                f"{reducer_name.upper()} + {classifier.upper()}"
            )
            metrics['reducer'] = reducer_name
            metrics['classifier'] = classifier
            results.append(metrics)
    
    # Compare results
    results_df = pd.DataFrame(results)
    print("\n=== Comparison of Dimensionality Reduction Techniques ===")
    print(results_df[['reducer', 'classifier', 'accuracy', 'train_time', 'predict_time']])
    
    # Plot comparison
    plt.figure(figsize=(12, 6))
    
    # Accuracy comparison
    plt.subplot(1, 2, 1)
    sns.barplot(x='reducer', y='accuracy', data=results_df)
    plt.title('Accuracy by Dimensionality Reduction Technique')
    plt.ylim(0, 1.0)
    
    # Training time comparison
    plt.subplot(1, 2, 2)
    sns.barplot(x='reducer', y='train_time', data=results_df)
    plt.title('Training Time by Dimensionality Reduction Technique')
    
    plt.tight_layout()
    plt.show()
    
    return results_df

def hyperparameter_tuning(X, y, reducer_name='pca', n_components=2, classifier='rf'):
    """Perform hyperparameter tuning for the specified pipeline."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Build the pipeline
    pipeline, reducer_name, classifier = build_pipeline(reducer_name, n_components, classifier)
    
    # Define parameter grid based on the classifier
    if classifier == 'rf':
        param_grid = {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__max_depth': [None, 10, 20, 30],
            'classifier__min_samples_split': [2, 5, 10],
        }
    elif classifier == 'svm':
        param_grid = {
            'classifier__C': [0.1, 1, 10],
            'classifier__kernel': ['linear', 'rbf'],
            'classifier__gamma': ['scale', 'auto'],
        }
    elif classifier == 'lr':
        param_grid = {
            'classifier__C': [0.1, 1, 10],
            'classifier__penalty': ['l1', 'l2'],
            'classifier__solver': ['liblinear'],
        }
    
    # Add reducer parameters if needed
    if reducer_name == 'pca':
        param_grid['reducer__n_components'] = [2, 5, 10, 0.95]  # 0.95 means 95% variance explained
    elif reducer_name == 'umap':
        param_grid['reducer__n_neighbors'] = [5, 15, 30]
        param_grid['reducer__min_dist'] = [0.1, 0.5, 0.99]
    
    # Perform grid search
    print(f"\n=== Hyperparameter Tuning for {reducer_name.upper()} + {classifier.upper()} ===")
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=5, n_jobs=-1, scoring='accuracy', verbose=1
=======
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
>>>>>>> student-branch
    )
    
    grid_search.fit(X_train, y_train)
    
<<<<<<< HEAD
    # Print best parameters and score
    print("\nBest parameters:")
    for param, value in grid_search.best_params_.items():
        print(f"{param}: {value}")
    
    print(f"\nBest cross-validation accuracy: {grid_search.best_score_:.4f}")
    
    # Evaluate on test set
    y_pred = grid_search.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest set accuracy: {test_accuracy:.4f}")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return grid_search

def main():
    # Load a dataset
    X, y, _ = load_dataset('digits')  # Try 'olivetti' or 'wine' for different datasets
    
    # 1. Compare different dimensionality reduction techniques
    print("\n=== Comparing Dimensionality Reduction Techniques ===")
    results = compare_reducers(X, y, reducers=['pca', 'umap'], classifier='rf')
    
    # 2. Hyperparameter tuning for the best pipeline
    best_reducer = results.loc[results['accuracy'].idxmax(), 'reducer']
    print(f"\n=== Tuning {best_reducer.upper()} with Random Forest ===")
    best_model = hyperparameter_tuning(X, y, reducer_name=best_reducer, classifier='rf')
    
    # 3. Compare different classifiers with the best reducer
    print("\n=== Comparing Classifiers with Best Reducer ===")
    classifiers = ['rf', 'svm', 'lr']
    
    all_results = []
    for clf in classifiers:
        print(f"\n=== Evaluating {best_reducer.upper()} with {clf.upper()} ===")
        pipeline, _, _ = build_pipeline(best_reducer, n_components=2, classifier=clf)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        metrics = evaluate_model(
            pipeline, X_train, X_test, y_train, y_test,
            f"{best_reducer.upper()} + {clf.upper()}"
        )
        metrics['classifier'] = clf
        all_results.append(metrics)
    
    # Plot classifier comparison
    all_results_df = pd.DataFrame(all_results)
    
    plt.figure(figsize=(12, 5))
    
    # Accuracy comparison
    plt.subplot(1, 2, 1)
    sns.barplot(x='classifier', y='accuracy', data=all_results_df)
    plt.title(f'Accuracy by Classifier (with {best_reducer.upper()})')
    plt.ylim(0, 1.0)
    
    # Training time comparison
    plt.subplot(1, 2, 2)
    sns.barplot(x='classifier', y='train_time', data=all_results_df)
    plt.title('Training Time by Classifier')
    
    plt.tight_layout()
    plt.show()
=======
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
>>>>>>> student-branch

if __name__ == "__main__":
    main()
