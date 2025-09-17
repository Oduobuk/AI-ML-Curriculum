"""
Exercise 2: t-SNE and UMAP for Non-linear Dimensionality Reduction

In this exercise, you'll explore t-SNE and UMAP for visualizing high-dimensional data.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.manifold import TSNE
import umap.umap_ as umap
from sklearn.datasets import fetch_openml
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import time

# Set random seed for reproducibility
np.random.seed(42)

def load_mnist(n_samples=5000):
    """Load and preprocess a subset of the MNIST dataset."""
    print("Loading MNIST dataset...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    
    # Take a random subset for faster computation
    indices = np.random.permutation(len(mnist.data))[:n_samples]
    X = mnist.data[indices]
    y = mnist.target[indices].astype(int)
    
    # Normalize pixel values to [0, 1]
    X = X / 255.0
    
    print(f"Loaded {X.shape[0]} samples with {X.shape[1]} features each.")
    return X, y

def plot_embedding(X_embedded, y, title, cmap='tab10'):
    """Plot 2D or 3D embeddings with class labels."""
    if X_embedded.shape[1] == 2:
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], 
                            c=y, cmap=cmap, alpha=0.7, s=10)
        plt.colorbar(scatter, label='Class')
        plt.xlabel('First Dimension')
        plt.ylabel('Second Dimension')
    elif X_embedded.shape[1] == 3:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(X_embedded[:, 0], X_embedded[:, 1], X_embedded[:, 2],
                           c=y, cmap=cmap, alpha=0.7, s=10)
        plt.colorbar(scatter, label='Class')
        ax.set_xlabel('First Dimension')
        ax.set_ylabel('Second Dimension')
        ax.set_zlabel('Third Dimension')
    
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def explore_tsne_parameters(X, y, perplexities=[5, 30, 50, 100], n_components=2):
    """Explore the effect of different perplexity values on t-SNE."""
    print("\n=== Exploring t-SNE with different perplexity values ===")
    
    plt.figure(figsize=(15, 10))
    for i, perplexity in enumerate(perplexities, 1):
        start_time = time.time()
        tsne = TSNE(n_components=n_components, 
                    perplexity=perplexity,
                    random_state=42,
                    n_iter=1000,
                    learning_rate='auto',
                    init='pca')
        
        print(f"Fitting t-SNE with perplexity={perplexity}...")
        X_tsne = tsne.fit_transform(X)
        
        plt.subplot(2, 2, i)
        scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, 
                            cmap='tab10', alpha=0.7, s=10)
        plt.title(f't-SNE (perplexity={perplexity})\nTime: {time.time() - start_time:.1f}s')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return X_tsne

def compare_tsne_umap(X, y, n_components=2):
    """Compare t-SNE and UMAP on the same dataset."""
    print("\n=== Comparing t-SNE and UMAP ===")
    
    # t-SNE
    print("Running t-SNE...")
    start_time = time.time()
    tsne = TSNE(n_components=n_components, 
                random_state=42,
                n_iter=1000,
                learning_rate='auto',
                init='pca')
    X_tsne = tsne.fit_transform(X)
    print(f"t-SNE completed in {time.time() - start_time:.1f} seconds")
    
    # UMAP
    print("\nRunning UMAP...")
    start_time = time.time()
    reducer = umap.UMAP(n_components=n_components,
                       random_state=42,
                       n_neighbors=15,
                       min_dist=0.1,
                       metric='euclidean')
    X_umap = reducer.fit_transform(X)
    print(f"UMAP completed in {time.time() - start_time:.1f} seconds")
    
    # Plot results
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    scatter1 = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, 
                          cmap='tab10', alpha=0.7, s=10)
    plt.colorbar(scatter1, label='Class')
    plt.title('t-SNE')
    plt.xlabel('First Dimension')
    plt.ylabel('Second Dimension')
    
    plt.subplot(1, 2, 2)
    scatter2 = plt.scatter(X_umap[:, 0], X_umap[:, 1], c=y, 
                          cmap='tab10', alpha=0.7, s=10)
    plt.colorbar(scatter2, label='Class')
    plt.title('UMAP')
    plt.xlabel('First Dimension')
    plt.ylabel('Second Dimension')
    
    plt.tight_layout()
    plt.show()
    
    return X_tsne, X_umap

def explore_umap_parameters(X, y, n_neighbors_list=[5, 15, 30, 50], min_dist=0.1):
    """Explore the effect of different n_neighbors values on UMAP."""
    print("\n=== Exploring UMAP with different n_neighbors values ===")
    
    plt.figure(figsize=(15, 10))
    for i, n_neighbors in enumerate(n_neighbors_list, 1):
        start_time = time.time()
        reducer = umap.UMAP(n_components=2,
                           n_neighbors=n_neighbors,
                           min_dist=min_dist,
                           random_state=42)
        
        print(f"Fitting UMAP with n_neighbors={n_neighbors}...")
        X_umap = reducer.fit_transform(X)
        
        plt.subplot(2, 2, i)
        scatter = plt.scatter(X_umap[:, 0], X_umap[:, 1], c=y, 
                            cmap='tab10', alpha=0.7, s=10)
        plt.title(f'UMAP (n_neighbors={n_neighbors})\nTime: {time.time() - start_time:.1f}s')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return X_umap

def visualize_3d_embeddings(X, y, method='umap'):
    """Create 3D visualizations of the embeddings."""
    print(f"\n=== Creating 3D {method.upper()} visualization ===")
    
    start_time = time.time()
    
    if method.lower() == 'tsne':
        reducer = TSNE(n_components=3, random_state=42, n_iter=1000)
    else:  # UMAP
        reducer = umap.UMAP(n_components=3, random_state=42)
    
    print(f"Fitting {method.upper()} in 3D...")
    X_3d = reducer.fit_transform(X)
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(X_3d[:, 0], X_3d[:, 1], X_3d[:, 2],
                       c=y, cmap='tab10', alpha=0.7, s=10)
    
    plt.colorbar(scatter, label='Class')
    ax.set_title(f'{method.upper()} 3D Projection\nTime: {time.time() - start_time:.1f}s')
    ax.set_xlabel('First Dimension')
    ax.set_ylabel('Second Dimension')
    ax.set_zlabel('Third Dimension')
    
    plt.tight_layout()
    plt.show()
    
    return X_3d

def main():
    # Load a sample dataset (MNIST)
    X, y = load_mnist(n_samples=2000)  # Using fewer samples for faster computation
    
    # 1. Explore t-SNE with different perplexity values
    explore_tsne_parameters(X, y, perplexities=[5, 30, 50, 100])
    
    # 2. Compare t-SNE and UMAP
    X_tsne, X_umap = compare_tsne_umap(X, y)
    
    # 3. Explore UMAP with different n_neighbors values
    explore_umap_parameters(X, y, n_neighbors_list=[5, 15, 30, 50])
    
    # 4. Create 3D visualizations
    visualize_3d_embeddings(X, y, method='tsne')
    visualize_3d_embeddings(X, y, method='umap')
    
    # 5. Interactive visualization with bokeh (optional)
    try:
        import bokeh.plotting as bp
        from bokeh.models import HoverTool, ColumnDataSource
        from bokeh.palettes import d3
        
        print("\nCreating interactive visualization with Bokeh...")
        
        # Use UMAP for the interactive visualization (faster than t-SNE)
        reducer = umap.UMAP(random_state=42)
        X_umap = reducer.fit_transform(X)
        
        # Create a Bokeh plot
        p = bp.figure(width=800, height=600, title="UMAP projection")
        
        # Create a color palette
        colors = d3["Category20"][10]
        
        # Add data points
        source = ColumnDataSource(data=dict(
            x=X_umap[:, 0],
            y=X_umap[:, 1],
            colors=[colors[i % 10] for i in y],
            label=[str(i) for i in y]
        ))
        
        p.scatter('x', 'y', color='colors', source=source, size=5, alpha=0.7)
        
        # Add hover tool
        hover = HoverTool(tooltips=[
            ("Label", "@label"),
            ("(x, y)", "($x, $y)")
        ])
        p.add_tools(hover)
        
        # Show the plot
        bp.output_notebook()
        bp.show(p)
        
    except ImportError:
        print("Bokeh not installed. Install with: pip install bokeh")

if __name__ == "__main__":
    main()
