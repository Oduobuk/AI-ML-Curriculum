# Month 4, Week 4 Exercises: NLP with Deep Learning

## Objective

These exercises are designed to test your understanding of word embeddings, their properties, and their role in modern Natural Language Processing (NLP) tasks using deep learning.

---

### Exercise 1: Word Embeddings vs. One-Hot Encoding

Consider the words "cat", "dog", and "table".

1.  **One-Hot Encoding:**
    *   How would these words be represented using one-hot encoding in a vocabulary of, say, 10,000 words?
    *   What is the mathematical relationship (e.g., dot product, cosine similarity) between the one-hot vectors for "cat" and "dog"? What about "cat" and "table"?
    *   What limitation of one-hot encoding does this highlight?

2.  **Word Embeddings:**
    *   Conceptually, how would the word embeddings for "cat" and "dog" be positioned in a 300-dimensional vector space relative to each other?
    *   How would the word embedding for "table" be positioned relative to "cat" and "dog"?
    *   What advantage do word embeddings offer in capturing semantic relationships that one-hot encoding cannot?

---

### Exercise 2: The Distributional Hypothesis

1.  Explain the "Distributional Hypothesis" in your own words.
2.  How does this hypothesis form the theoretical basis for learning word embeddings like Word2Vec?
3.  Provide a simple example of two words that you would expect to have similar contexts and thus similar word embeddings.

---

### Exercise 3: Word2Vec Architectures

Briefly describe the core difference between the **Skip-gram** and **CBOW (Continuous Bag-of-Words)** architectures in Word2Vec. What is each model trying to predict?

---

### Exercise 4: Interpreting Word Embeddings

Imagine you have trained word embeddings and find the following vector relationships:

*   `vector("Paris") - vector("France") + vector("Italy") ≈ vector("Rome")`
*   `vector("walking") - vector("walk") + vector("swimming") ≈ vector("swim")`

1.  What kind of semantic relationship is captured in the first example?
2.  What kind of semantic relationship is captured in the second example?
3.  Why is the ability to capture such relationships a significant advancement for NLP?

---

### Exercise 5: Deep Learning in NLP

1.  Before the widespread adoption of deep learning, how were features typically engineered for NLP tasks (e.g., for named entity recognition)?
2.  How has deep learning changed this approach, particularly with the use of word embeddings and recurrent/convolutional layers?
3.  What is the main benefit of this shift from hand-engineered features to learned representations?
