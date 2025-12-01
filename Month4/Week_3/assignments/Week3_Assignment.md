# Month 4, Week 3 Assignment: Designing a Recurrent Neural Network

## Objective

This assignment requires you to design a simple Recurrent Neural Network (RNN), Long Short-Term Memory (LSTM), or Gated Recurrent Unit (GRU) architecture for a sequence prediction task. You will outline your approach, specifying layer types, input/output shapes, and activation functions, without writing any code. This exercise focuses on understanding how these networks process sequential data.

## Scenario

You are tasked with building a model for **Sentiment Analysis** of movie reviews. The input will be a sequence of words (represented as numerical embeddings), and the output should be a classification of the sentiment (positive, negative, or neutral).

**Input Data Characteristics:**
*   Each word is represented by a 100-dimensional word embedding vector.
*   Reviews can vary in length, but you decide to pad/truncate all sequences to a maximum length of 50 words.
*   The vocabulary size is 10,000 unique words.

**Output Task:**
*   Classify the sentiment into 3 categories: Positive, Negative, Neutral.

## Instructions

Describe a simple recurrent neural network architecture layer by layer. For each layer, you must specify its type, key parameters, and the shape of its output.

---

### Your Proposed Architecture

Fill out the details for each layer below.

#### Layer 1: Input Layer (Embedding)

*   **Type:** Embedding Layer (conceptually, this is where your word embeddings come in)
*   **Parameters:**
    *   Vocabulary Size: `10,000`
    *   Embedding Dimension: `100`
    *   Input Sequence Length: `50`
*   **Output Shape:** `(batch_size, 50, 100)` (where 50 is sequence length, 100 is embedding dim)

---

#### Layer 2: Recurrent Layer (Choose one: Simple RNN, LSTM, or GRU)

*   **Type:** `[Choose one: Simple RNN, LSTM, or GRU]`
*   **Parameters:**
    *   Number of Units (Hidden State Dimension): `64`
    *   `return_sequences`: `False` (Hint: You only need the final output for classification)
*   **Output Shape:** `(batch_size, 64)` (if `return_sequences=False`)

    *   **Justification for your choice of RNN/LSTM/GRU:** Briefly explain why you chose this specific type of recurrent layer for sentiment analysis.

---

#### Layer 3: Dense Layer

*   **Type:** Fully Connected (Dense) Layer
*   **Parameters:**
    *   Number of Neurons: `32`
    *   Activation Function: `ReLU`
*   **Output Shape:** `(batch_size, 32)`

---

#### Layer 4: Output Layer

*   **Type:** Fully Connected (Dense) Layer
*   **Parameters:**
    *   Number of Neurons: `?` (Hint: How many sentiment classes?)
    *   Activation Function: `?` (Hint: For multi-class classification, what activation function outputs probabilities?)
*   **Output Shape:** `(batch_size, ?)`

## Submission

*   Complete the details for each layer, filling in the `?` marks and making your choice for the recurrent layer.
*   Save your architecture design in a Markdown file named `Week3_RNN_Architecture_Design.md`.
*   Be prepared to discuss your design choices and the rationale behind them.
