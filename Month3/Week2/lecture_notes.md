# Week 2: Dimensionality Reduction & Feature Engineering

## 1. Introduction to Dimensionality Reduction

### 1.1 The Curse of Dimensionality
- As the number of features (dimensions) increases, the data becomes increasingly sparse
- Distance measures become less meaningful in high dimensions
- The volume of space increases exponentially with dimensions
- More dimensions = more data needed for meaningful analysis

### 1.2 Why Dimensionality Reduction?
- Reduce computational cost
- Remove noise and redundant features
- Visualize high-dimensional data
- Improve model performance
- Avoid overfitting

## 2. Principal Component Analysis (PCA)

### 2.1 Intuition
- Projects data onto principal components that maximize variance
- First principal component captures most variance, second (orthogonal to first) captures next most, etc.

### 2.2 Mathematics of PCA
1. Standardize the data
2. Compute covariance matrix
3. Calculate eigenvalues and eigenvectors
4. Sort eigenvalues in descending order
5. Select top k eigenvectors
6. Transform original data

### 2.3 Choosing Number of Components
- Scree plot (elbow method)
- Cumulative explained variance (e.g., 95% variance)
- Cross-validation

## 3. t-SNE (t-Distributed Stochastic Neighbor Embedding)

### 3.1 Key Concepts
- Non-linear dimensionality reduction
- Preserves local similarities
- Good for visualization (2D/3D)
- Perplexity parameter controls neighborhood size

### 3.2 How t-SNE Works
1. Computes probabilities that represent pairwise similarities
2. Constructs similar probability distribution in lower dimension
3. Minimizes Kullback-Leibler divergence between distributions

### 3.3 Strengths and Weaknesses
**Strengths:**
- Captures non-linear relationships
- Preserves local structure
- Effective for visualization

**Weaknesses:**
- Computationally expensive
- Perplexity parameter needs tuning
- Results can vary between runs

## 4. UMAP (Uniform Manifold Approximation and Projection)

### 4.1 Key Concepts
- Non-linear dimensionality reduction
- Preserves both local and global structure
- Faster than t-SNE for large datasets
- Based on manifold learning and topological data analysis

### 4.2 How UMAP Works
1. Constructs a high-dimensional graph
2. Optimizes a low-dimensional graph to be as similar as possible
3. Uses stochastic gradient descent for optimization

### 4.3 Comparison with t-SNE
- UMAP is generally faster
- UMAP often preserves more global structure
- t-SNE might be better for local structure
- UMAP has fewer parameters to tune

## 5. Feature Selection Techniques

### 5.1 Filter Methods
- Select features based on statistical tests
- Examples: Variance threshold, chi-square test, mutual information
- Fast and scalable

### 5.2 Wrapper Methods
- Use a model to evaluate feature subsets
- Examples: Recursive Feature Elimination (RFE)
- Computationally expensive

### 5.3 Embedded Methods
- Perform feature selection during model training
- Examples: Lasso regression, Random Forest feature importance
- Balance between filter and wrapper methods

## 6. Building ML Pipelines

### 6.1 Why Pipelines?
- Streamline machine learning workflows
- Prevent data leakage
- Make code more readable and reproducible
- Simplify model deployment

### 6.2 Pipeline Components
1. Data preprocessing (scaling, encoding)
2. Feature selection/extraction
3. Model training
4. Model evaluation

### 6.3 Example Pipeline
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=0.95)),  # Keep 95% of variance
    ('classifier', RandomForestClassifier())
])
```

## 7. Practical Considerations

### 7.1 When to Use Dimensionality Reduction
- Visualizing high-dimensional data
- Speeding up model training
- Reducing noise in the data
- When features are highly correlated

### 7.2 Common Pitfalls
- Losing interpretability of features
- Choosing wrong number of components
- Applying linear methods to non-linear data
- Not scaling data before PCA

### 7.3 Best Practices
- Always scale your data before PCA
- Visualize your data in 2D/3D
- Use cross-validation to evaluate impact
- Consider both linear and non-linear methods

## 8. Case Studies

### 8.1 Image Compression with PCA
- How PCA can be used for image compression
- Trade-off between compression and quality
- Implementation example with scikit-learn

### 8.2 Visualizing Word Embeddings
- Using t-SNE/UMAP to visualize word vectors
- Understanding semantic relationships
- Interactive visualization with Plotly

## 9. Advanced Topics

### 9.1 Kernel PCA
- Non-linear extension of PCA
- Uses kernel trick to handle non-linear relationships
- More powerful but more computationally expensive

### 9.2 Autoencoders for Dimensionality Reduction
- Neural network approach
- Can learn non-linear transformations
- More flexible but requires more data

## 10. Resources
- [Scikit-learn Dimensionality Reduction](https://scikit-learn.org/stable/modules/unsupervised_reduction.html)
- [UMAP Documentation](https://umap-learn.readthedocs.io/)
- [Distill.pub - How to Use t-SNE Effectively](https://distill.pub/2016/misread-tsne/)
