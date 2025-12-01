# Month 4, Week 3 Exercises: RNNs, LSTMs, and GRUs

## Objective

These exercises are designed to test your conceptual understanding of Recurrent Neural Networks (RNNs), Long Short-Term Memory (LSTMs), and Gated Recurrent Units (GRUs), their mechanisms, and their applications to sequential data.

---

### Exercise 1: Identifying Appropriate Architectures

For each scenario below, suggest whether a **Simple RNN**, an **LSTM**, or a **GRU** would be the most appropriate choice. Justify your answer in 1-2 sentences, considering the nature of the data and the task.

1.  **Scenario A:** Predicting the next word in a sentence, where the context for prediction might span many words (e.g., understanding the subject-verb agreement across a long clause).
    *   **Choice:**
    *   **Justification:**

2.  **Scenario B:** Classifying short, simple sequences of digits (e.g., recognizing a 3-digit code), where long-term dependencies are not a major concern.
    *   **Choice:**
    *   **Justification:**

3.  **Scenario C:** Forecasting stock prices based on historical data, where both recent trends and long-term economic indicators might influence future prices. You need a model that is computationally efficient to train on a large dataset.
    *   **Choice:**
    *   **Justification:**

4.  **Scenario D:** Generating complex musical compositions, where the structure and harmony need to be consistent over very long stretches of music.
    *   **Choice:**
    *   **Justification:**

---

### Exercise 2: The Vanishing Gradient Problem

1.  Briefly explain (1-2 sentences) what the **vanishing gradient problem** is in the context of traditional RNNs.
2.  How does this problem affect the ability of an RNN to learn from sequential data?
3.  Name the primary architectural component in LSTMs that was designed to mitigate this problem.

---

### Exercise 3: LSTM Gate Functions

Describe the primary role of each of the following gates within an LSTM cell:

1.  **Forget Gate:**
2.  **Input Gate:**
3.  **Output Gate:**

---

### Exercise 4: LSTM vs. GRU

1.  What is the main architectural difference between an LSTM and a GRU in terms of their gates?
2.  In what situations might you prefer to use a GRU over an LSTM, and vice-versa?

---

### Exercise 5: Unrolling an RNN

Imagine a simple RNN processing the sentence "The cat sat on the mat."

1.  How many time steps would this RNN be unrolled for (assuming each word is one time step)?
2.  At each time step, what two pieces of information are typically fed into the RNN cell?
3.  What is the primary advantage of "unrolling" the RNN for understanding its operation?
