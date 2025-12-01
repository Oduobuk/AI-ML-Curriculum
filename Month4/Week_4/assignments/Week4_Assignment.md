# Month 4, Week 4 Assignment: Word Embeddings and Text Classification

## Objective

This assignment aims to deepen your understanding of word embeddings, their ability to capture semantic relationships, and how they are integrated into deep learning models for Natural Language Processing tasks. You will describe a text classification pipeline using pre-trained word embeddings.

## Instructions

In a Markdown file, address the following points:

### 1. Word Embeddings and Semantic Relationships

*   **Explain the core idea behind word embeddings.** How do they differ from one-hot encodings in representing words?
*   **Describe how word embeddings capture semantic similarity.** Provide an example of a semantic relationship (e.g., King - Man + Woman = Queen) and explain how this relationship is represented in the vector space.
*   **Discuss the advantages of using word embeddings** over traditional sparse representations (like one-hot vectors) for NLP tasks.

### 2. Text Classification Pipeline with Pre-trained Embeddings

Imagine you are building a model to classify news articles into categories (e.g., "Sports", "Politics", "Technology"). You have access to a dataset of news articles and their corresponding labels. You also decide to use **pre-trained word embeddings** (e.g., Word2Vec, GloVe) to initialize your model.

Outline a simple deep learning pipeline for this text classification task, describing each component:

#### a) Input Layer (Text Preprocessing & Embedding Lookup)

*   How would you convert raw text (sentences) into a sequence of numerical inputs suitable for a neural network?
*   How would you use the pre-trained word embeddings in this step?

#### b) Feature Extraction Layer (Recurrent or Convolutional)

*   Which type of neural network layer would you choose to process the sequence of word embeddings and extract higher-level features from the text? (e.g., LSTM, GRU, 1D CNN).
*   Briefly explain why this layer is suitable for capturing patterns in text sequences.
*   Specify a conceptual output shape for this layer.

#### c) Classification Head (Dense Layers)

*   After extracting features, how would you transition to making a classification decision?
*   Describe the type and number of layers you would use in this part of the pipeline.
*   What activation function would you use in the final output layer, and why?

#### d) Output

*   What would be the final output of your model?

## Submission

*   Save your detailed explanation and pipeline design in a Markdown file named `Week4_NLP_Assignment.md`.
*   Be prepared to discuss your design choices.
