# Week 2 Project: Dimensionality Reduction for Image Classification

## 📋 Project Overview
In this project, you'll apply dimensionality reduction techniques to an image classification task. You'll work with the Fashion MNIST dataset, which consists of 70,000 grayscale images of 10 different fashion items.

## 🎯 Learning Objectives
- Apply PCA, t-SNE, and UMAP to high-dimensional image data
- Compare the performance of different dimensionality reduction techniques
- Build and evaluate classifiers on reduced feature spaces
- Analyze the trade-offs between dimensionality reduction and model performance

## 📂 Project Structure
```
project/
├── data/                    # Dataset (will be downloaded)
├── notebooks/               # Jupyter notebooks for analysis
│   ├── 1_exploratory_analysis.ipynb
│   ├── 2_dimensionality_reduction.ipynb
│   └── 3_modeling.ipynb
├── src/                     # Source code
│   ├── data_loading.py
│   ├── preprocessing.py
│   ├── dimensionality_reduction.py
│   └── modeling.py
├── requirements.txt         # Project dependencies
└── README.md               # This file
```

## 🚀 Getting Started

### Prerequisites
- Python 3.7+
- Required packages (install with `pip install -r requirements.txt`):
  ```
  numpy
  pandas
  matplotlib
  seaborn
  scikit-learn
  umap-learn
  jupyter
  ```

### Dataset
We'll use the Fashion MNIST dataset, which will be automatically downloaded when you run the code.

## 📝 Tasks

### 1. Exploratory Data Analysis (EDA)
- Load and visualize sample images from the dataset
- Analyze the distribution of classes
- Examine basic statistics of the data

### 2. Dimensionality Reduction
- Implement PCA and analyze explained variance
- Apply t-SNE and UMAP for visualization
- Compare the results of different techniques

### 3. Classification with Dimensionality Reduction
- Train a baseline classifier on the original data
- Train classifiers on reduced feature spaces
- Compare model performance and training time
- Analyze the trade-offs between accuracy and dimensionality

### 4. Advanced Analysis (Bonus)
- Experiment with different numbers of components
- Try combining multiple dimensionality reduction techniques
- Visualize decision boundaries
- Implement and compare with autoencoders

## 📊 Expected Deliverables
1. A Jupyter notebook with your analysis and visualizations
2. A report (PDF or Markdown) summarizing your findings
3. Source code for any custom implementations
4. A presentation (5-7 slides) highlighting key insights

## 📅 Timeline
- **Day 1-2**: Data exploration and preprocessing
- **Day 3-4**: Implement and compare dimensionality reduction techniques
- **Day 5**: Model training, evaluation, and final presentation

## 📚 Resources
- [Fashion MNIST Dataset](https://github.com/zalandoresearch/fashion-mnist)
- [Scikit-learn Dimensionality Reduction](https://scikit-learn.org/stable/modules/unsupervised_reduction.html)
- [UMAP Documentation](https://umap-learn.readthedocs.io/)
- [t-SNE Explained](https://distill.pub/2016/misread-tsne/)

## 🎯 Evaluation Criteria
- Code quality and organization (30%)
- Analysis depth and insights (30%)
- Visualization quality (20%)
- Presentation and documentation (20%)
