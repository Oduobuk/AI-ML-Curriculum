# Month 4, Week 4 Cheatsheet: NLP with Deep Learning

## 1. Natural Language Processing (NLP)

*   **Definition:** The field focused on enabling computers to understand, interpret, and generate human language.
*   **Challenges:**
    *   **Ambiguity:** Words/phrases can have multiple meanings.
    *   **Context-Dependency:** Meaning is heavily influenced by surrounding text and real-world knowledge.
    *   **Sparsity:** Vast vocabularies lead to many rare words.
    *   **Efficiency through Omission:** Humans often omit information, expecting inference.

---

## 2. Representing Words

### a) One-Hot Encoding (Traditional)

*   **Concept:** Each word is a unique vector with a `1` at its index and `0`s elsewhere.
*   **Limitations:**
    *   **High Dimensionality:** Vector size equals vocabulary size (millions for large vocabularies).
    *   **No Semantic Relationship:** All words are equidistant (orthogonal); "cat" is as similar to "dog" as it is to "banana".

### b) Word Embeddings (Modern Deep Learning)

*   **Concept:** Words are represented as dense, low-dimensional, real-valued vectors (e.g., 50-1000 dimensions).
*   **Distributional Hypothesis:** "You shall know a word by the company it keeps." (J.R. Firth). Words appearing in similar contexts have similar meanings.
*   **Key Idea:** Semantic similarity is captured by vector proximity in the embedding space.
    *   `vector("king") - vector("man") + vector("woman") ≈ vector("queen")`
*   **Advantages:**
    *   **Captures Semantic Relationships:** Similar words are close in vector space.
    *   **Reduced Dimensionality:** More efficient representation.
    *   **Learned Automatically:** From large text corpora.

---

## 3. Word2Vec Algorithm

A popular and efficient algorithm for learning word embeddings.

### a) Skip-gram Model

*   **Goal:** Predict **context words** given a **target word**.
*   **Mechanism:** For each word in the corpus, it tries to predict the words within a fixed-size window around it.
*   **Intuition:** A word's meaning is defined by the words it co-occurs with.

### b) Continuous Bag-of-Words (CBOW) Model

*   **Goal:** Predict a **target word** given its **context words**.
*   **Mechanism:** Takes the average of the word vectors of the context words to predict the target word.
*   **Intuition:** The context provides strong clues about the missing word.

---

## 4. Deep Learning in NLP Pipeline (General)

1.  **Text Preprocessing:** Tokenization, lowercasing, removing stop words, etc.
2.  **Word Embedding Layer:** Converts words (or sub-word units) into dense vector representations. Can use pre-trained embeddings (e.g., Word2Vec, GloVe, FastText) or learn them from scratch.
3.  **Sequence Modeling Layer:** Processes the sequence of embeddings to capture contextual information.
    *   **RNNs/LSTMs/GRUs:** Excellent for capturing long-range dependencies in sequences.
    *   **1D CNNs:** Can capture local patterns (n-grams) in text.
    *   **Transformers:** (More advanced, covered in later courses) State-of-the-art for many NLP tasks, relying on self-attention mechanisms.
4.  **Pooling/Flattening:** Reduces the sequence of hidden states to a fixed-size vector representation of the entire input (e.g., global max pooling, taking the last hidden state).
5.  **Dense (Fully Connected) Layers:** For classification or regression tasks.
6.  **Output Layer:**
    *   **Softmax:** For multi-class classification (e.g., sentiment analysis).
    *   **Sigmoid:** For binary classification or multi-label classification.

---

## 5. Key Concepts

*   **Corpus:** A large collection of text data.
*   **Vocabulary:** The set of all unique words in a corpus.
*   **Tokenization:** Breaking down text into individual words or sub-word units (tokens).
*   **Word Embeddings:** Dense vector representations of words.
*   **Pre-trained Embeddings:** Word embeddings learned from very large corpora (e.g., Common Crawl, Wikipedia) and made publicly available. These are often used as a starting point for new NLP tasks.
