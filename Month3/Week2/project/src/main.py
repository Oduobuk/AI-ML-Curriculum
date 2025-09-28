"""
Main script for the Fashion MNIST classification project.
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from data_loading import load_fashion_mnist, get_class_names
from preprocessing import create_preprocessing_pipeline, apply_preprocessing
from modeling import train_random_forest, evaluate_model

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Fashion MNIST Classification')
    parser.add_argument('--n_components', type=int, default=None,
                       help='Number of PCA components (default: None)')
    parser.add_argument('--n_estimators', type=int, default=100,
                       help='Number of trees in the random forest')
    parser.add_argument('--max_depth', type=int, default=None,
                       help='Maximum depth of the trees')
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()
    
    # Load data
    print("Loading data...")
    X_train, X_test, y_train, y_test = load_fashion_mnist()
    
    # Preprocess data
    print("Preprocessing data...")
    pipeline = create_preprocessing_pipeline(n_components=args.n_components)
    X_train_transformed, X_test_transformed = apply_preprocessing(
        pipeline, X_train, X_test
    )
    
    # Print data shapes
    print(f"Training data shape: {X_train_transformed.shape}")
    print(f"Test data shape: {X_test_transformed.shape}")
    
    # Train model
    print("Training model...")
    model = train_random_forest(
        X_train_transformed, y_train,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth
    )
    
    # Evaluate model
    print("Evaluating model...")
    metrics = evaluate_model(model, X_test_transformed, y_test)
    
    # Print results
    print(f"\nTest Accuracy: {metrics['accuracy']:.4f}")
    print("\nClassification Report:")
    print(metrics['report'])
    
    # Plot feature importances if using Random Forest without PCA
    if hasattr(model, 'feature_importances_') and args.n_components is None:
        plt.figure(figsize=(10, 6))
        plt.bar(range(50), model.feature_importances_[:50])
        plt.title('Feature Importances (First 50 Features)')
        plt.xlabel('Feature Index')
        plt.ylabel('Importance')
        plt.tight_layout()
        plt.savefig('feature_importances.png')
        plt.show()

if __name__ == "__main__":
    main()
