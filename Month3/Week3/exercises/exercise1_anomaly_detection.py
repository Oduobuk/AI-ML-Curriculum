"""
Exercise 1: Anomaly Detection

This exercise covers different anomaly detection techniques:
1. Statistical methods (Z-score, IQR)
2. Isolation Forest
3. One-class SVM
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

from data.datasets import generate_anomaly_data

def plot_anomalies(X, y_pred, title):
    """Plot normal points and anomalies."""
    plt.figure(figsize=(10, 6))
    
    # Plot normal points
    plt.scatter(
        X[y_pred == 1, 0], 
        X[y_pred == 1, 1],
        c='blue', 
        label='Normal'
    )
    
    # Plot anomalies
    plt.scatter(
        X[y_pred == -1, 0], 
        X[y_pred == -1, 1],
        c='red', 
        label='Anomaly'
    )
    
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def statistical_methods(X):
    """Detect anomalies using statistical methods."""
    # Z-score method
    z_scores = np.abs((X - X.mean(axis=0)) / X.std(axis=0))
    z_anomalies = np.any(z_scores > 3, axis=1)  # Points with |Z| > 3
    
    # IQR method
    q1 = np.percentile(X, 25, axis=0)
    q3 = np.percentile(X, 75, axis=0)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    iqr_anomalies = np.any((X < lower_bound) | (X > upper_bound), axis=1)
    
    return z_anomalies, iqr_anomalies

def main():
    # Generate sample data
    X, y_true = generate_anomaly_data()
    
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 1. Statistical methods
    z_anomalies, iqr_anomalies = statistical_methods(X_scaled)
    
    # 2. Isolation Forest
    iso_forest = IsolationForest(
        contamination=0.05, 
        random_state=42
    )
    iso_pred = iso_forest.fit_predict(X_scaled)
    
    # 3. One-class SVM
    oc_svm = OneClassSVM(
        nu=0.05,  # Expected proportion of outliers
        gamma='auto'
    )
    svm_pred = oc_svm.fit_predict(X_scaled)
    
    # Plot results
    plot_anomalies(X, z_anomalies, 'Z-score Anomaly Detection')
    plot_anomalies(X, iqr_anomalies, 'IQR Anomaly Detection')
    plot_anomalies(X, iso_pred, 'Isolation Forest')
    plot_anomalies(X, svm_pred, 'One-class SVM')
    
    # Print accuracy (for demonstration)
    print(f"Z-score accuracy: {np.mean(z_anomalies == (y_true == -1)):.2f}")
    print(f"IQR accuracy: {np.mean(iqr_anomalies == (y_true == -1)):.2f}")
    print(f"Isolation Forest accuracy: {np.mean(iso_pred == y_true):.2f}")
    print(f"One-class SVM accuracy: {np.mean(svm_pred == y_true):.2f}")

if __name__ == "__main__":
    main()
