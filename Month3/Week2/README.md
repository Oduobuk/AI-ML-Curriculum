# Week 2: Dimensionality Reduction

## 🎓 Learning Objectives

By the end of this week, you will be able to:
- Understand the curse of dimensionality and the need for dimensionality reduction
- Implement and apply Principal Component Analysis (PCA)
- Work with t-SNE and UMAP for non-linear dimensionality reduction
- Understand the mathematics behind SVD and its relationship to PCA
- Evaluate the effectiveness of dimensionality reduction techniques
- Apply these techniques to real-world datasets

## 📅 Weekly Schedule

### Day 1: Introduction to Dimensionality Reduction
- **Morning**: Understanding the curse of dimensionality
- **Afternoon**: Feature selection vs. feature extraction

### Day 2: Principal Component Analysis (PCA)
- **Morning**: Mathematics behind PCA
- **Afternoon**: Implementing PCA from scratch

### Day 3: Advanced Dimensionality Reduction
- **Morning**: t-SNE and UMAP for non-linear reduction
- **Afternoon**: Practical implementation and comparison

### Day 4: Applications and Case Studies
- **Morning**: Real-world applications
- **Afternoon**: Hands-on project work

### Day 5: Project Presentations
- **Morning**: Final project work
- **Afternoon**: Presentations and peer review

## 🛠️ Exercises

### Exercise 1: PCA Implementation
- Implement PCA from scratch using NumPy
- Apply to high-dimensional datasets
- Visualize explained variance ratio
- Compare with scikit-learn's implementation

### Exercise 2: t-SNE and UMAP
- Apply t-SNE to high-dimensional data
- Compare with UMAP results
- Tune perplexity and other parameters
- Visualize high-dimensional clusters in 2D/3D

### Exercise 3: Dimensionality Reduction Pipelines
- Build end-to-end ML pipelines with dimensionality reduction
- Evaluate impact on model performance
- Handle categorical variables
- Optimize preprocessing steps

## 📂 Project: High-Dimensional Data Analysis

### Overview
Apply dimensionality reduction techniques to a high-dimensional dataset (e.g., image data, gene expression data). Analyze how different techniques affect downstream tasks.

### Requirements:
1. Perform EDA on the high-dimensional dataset
2. Apply at least three different dimensionality reduction techniques
3. Evaluate and compare the results
4. Build a simple classifier on the reduced features
5. Document your findings and insights

### Deliverables:
- Jupyter notebook with complete analysis
- Presentation of key findings
- Performance comparison of different techniques

## 📚 Resources

### Core Reading
- [Scikit-learn Dimensionality Reduction](https://scikit-learn.org/stable/modules/decomposition.html)
- [Visualizing PCA](https://setosa.io/ev/principal-component-analysis/)
- [How to Use t-SNE Effectively](https://distill.pub/2016/misread-tsne/)

### Additional Materials
- [UMAP Documentation](https://umap-learn.readthedocs.io/)
- [Dimensionality Reduction for Machine Learning](https://machinelearningmastery.com/dimensionality-reduction-algorithms-with-python/)
- [Interactive PCA Visualization](https://projector.tensorflow.org/)

## 🧩 Prerequisites

### Technical Skills
- Python programming (NumPy, Pandas, scikit-learn)
- Basic linear algebra (eigenvalues, eigenvectors)
- Data visualization (Matplotlib, Seaborn)
- Experience with Jupyter Notebooks

### Software Requirements
- Python 3.8+
- Jupyter Lab/Notebook
- Required packages (see `requirements.txt`)

## 📊 Assessment

### Grading Breakdown
- **Exercise Submissions**: 40%
  - Code quality and documentation
  - Correctness of implementation
  - Quality of visualizations
  - Interpretation of results

- **Weekly Project**: 40%
  - Data preprocessing and exploration
  - Implementation and evaluation of techniques
  - Analysis and insights
  - Presentation quality

- **Participation**: 20%
  - Class discussions
  - Peer code reviews
  - Active engagement in group activities

## 🚀 Getting Started

1. Clone this repository:
   ```bash
   git clone [repository-url]
   cd Month3/Week2
   ```

2. Set up your environment:
   ```bash
   pip install -r requirements.txt
   ```

3. Start Jupyter Lab:
   ```bash
   jupyter lab
   ```

## 🤝 Support

For questions or assistance, please:
1. First check the [GitHub Issues](https://github.com/yourusername/ai-ml-curriculum/issues)
2. Post in the course discussion forum
3. Schedule office hours with the instructor

## 📝 License

This curriculum is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
