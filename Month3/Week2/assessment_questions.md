# Week 2 Assessment: Dimensionality Reduction & Feature Engineering

## Multiple Choice Questions (1 point each)

1. **What is the primary goal of Principal Component Analysis (PCA)?**
   a) To increase the number of features in the dataset  
   b) To reduce the dimensionality while preserving as much variance as possible  
   c) To classify data points into different categories  
   d) To increase the correlation between features

2. **In the context of t-SNE, what does the perplexity parameter control?**
   a) The number of dimensions in the output space  
   b) The balance between local and global aspects of the data  
   c) The learning rate of the optimization  
   d) The number of iterations for convergence

3. **Which of the following is NOT a benefit of dimensionality reduction?**
   a) Reduces computational requirements  
   b) Helps in data visualization  
   c) Always improves model accuracy  
   d) Reduces storage requirements

4. **What is the key difference between UMAP and t-SNE?**
   a) UMAP is faster and better preserves global structure  
   b) t-SNE can only be used for classification tasks  
   c) UMAP doesn't require normalization of input data  
   d) t-SNE is deterministic while UMAP is not

## Short Answer Questions (2 points each)

5. **Explain the concept of explained variance ratio in PCA and how it can be used to determine the optimal number of components.**

6. **Compare and contrast the computational complexity of PCA, t-SNE, and UMAP. When would you choose one over the others?**

7. **Describe a scenario where feature selection would be more appropriate than feature extraction techniques like PCA.**

## Practical Exercise (5 points)

8. **Dimensionality Reduction Implementation**
   - Load the digits dataset from scikit-learn
   - Apply PCA to reduce the dimensions to 2
   - Apply t-SNE with different perplexity values (5, 30, 50)
   - Create visualizations comparing the results
   - Write a brief analysis of your observations

## Advanced Topics (Bonus - 3 points)

9. **Autoencoders for Dimensionality Reduction**
   - Implement a simple autoencoder using Keras/TensorFlow
   - Train it on the MNIST dataset
   - Compare its performance with PCA in terms of reconstruction error
   - Discuss the advantages and disadvantages of using autoencoders versus traditional methods

## Submission Guidelines
- Submit your answers as a PDF document
- Include all code used for the practical exercises
- Clearly label all sections and questions
- Include any references used

## Grading Rubric
- Multiple Choice: 1 point per question (4 points total)
- Short Answer: 2 points per question (6 points total)
- Practical Exercise: 5 points (code: 2, visualization: 2, analysis: 1)
- Bonus: Up to 3 points for the advanced topic

**Total Points: 15 (18 with bonus)**
