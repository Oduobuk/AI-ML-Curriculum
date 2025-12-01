# Month 4, Week 3: Recurrent Neural Networks (RNNs), LSTMs, and GRUs

## 1. Introduction to Sequential Data and Recurrent Neural Networks (RNNs)

Many real-world datasets are sequential, meaning the order of data points carries crucial information. Examples include:
*   **Time Series Data:** Stock prices, weather patterns, sensor readings.
*   **Natural Language:** Sentences, paragraphs, documents (words appear in a specific order).
*   **Speech:** Audio signals (sequence of sound waves).
*   **Video:** A sequence of image frames.

Traditional feedforward neural networks (like the CNNs we've studied) treat each input independently. They lack "memory" and cannot effectively process variable-length sequences or capture dependencies between elements in a sequence.

**Recurrent Neural Networks (RNNs)** are specifically designed to handle sequential data. Their key characteristic is a **recurrent connection**, which allows information to persist from one step of the sequence to the next. This internal "hidden state" acts as a memory, enabling the network to learn patterns and dependencies across time steps.

**How RNNs Work (Conceptually):**
At each time step `t`, an RNN takes two inputs:
1.  The current input from the sequence, `x_t`.
2.  The hidden state from the previous time step, `h_{t-1}`.

It then computes a new hidden state, `h_t`, and an output, `y_t`. The hidden state `h_t` is a function of `x_t` and `h_{t-1}`. This process can be visualized by "unrolling" the RNN over time, showing how the same set of weights is applied at each step.

**Applications of RNNs:**
*   Machine Translation (e.g., Google Translate)
*   Speech Recognition (e.g., Siri, Alexa)
*   Time Series Prediction (e.g., stock market forecasting)
*   Text Generation (e.g., generating coherent sentences)
*   Sentiment Analysis

## 2. The Vanishing/Exploding Gradient Problem in RNNs

Training RNNs involves a technique called **Backpropagation Through Time (BPTT)**, which is essentially backpropagation applied to the unrolled network. However, traditional RNNs suffer from a significant challenge: the **vanishing or exploding gradient problem**.

*   **Vanishing Gradients:** As gradients are propagated backward through many time steps, they can become extremely small. This makes it difficult for the network to learn long-term dependencies, as the influence of early inputs on later outputs diminishes rapidly. The network "forgets" information from the distant past.
*   **Exploding Gradients:** Conversely, gradients can also become extremely large, leading to unstable training and large weight updates that prevent the model from converging.

This problem severely limits the ability of simple RNNs to model sequences where dependencies span many time steps.

## 3. Long Short-Term Memory (LSTM) Networks

**Long Short-Term Memory (LSTMs)** networks are a special kind of RNN designed to explicitly address the vanishing gradient problem and effectively learn long-term dependencies. They achieve this through a sophisticated internal mechanism called a **cell state** and several **gates**.

**Key Components of an LSTM Cell:**
*   **Cell State (`C_t`):** This is the "memory" of the LSTM. It runs straight through the entire chain, with only minor linear interactions. Information can be added to or removed from the cell state, allowing it to carry relevant information across long sequences.
*   **Gates:** These are neural network layers (typically sigmoid activation functions) that output values between 0 and 1. They act as "filters" or "switches" that control how much information flows through the cell.
    *   **Forget Gate (`f_t`):** Decides what information from the previous cell state (`C_{t-1}`) should be forgotten or discarded. A value of 0 means "forget completely," and 1 means "keep completely."
    *   **Input Gate (`i_t`) and Candidate Cell State (`\tilde{C}_t`):** The input gate decides which new information from the current input (`x_t`) and previous hidden state (`h_{t-1}`) is relevant. The candidate cell state (`\tilde{C}_t`) is a new potential memory that could be added.
    *   **Output Gate (`o_t`):** Decides what part of the current cell state (`C_t`) should be outputted as the new hidden state (`h_t`).

By carefully orchestrating these gates, LSTMs can selectively remember or forget information, making them highly effective at capturing long-range dependencies in sequential data.

## 4. Gated Recurrent Units (GRUs)

**Gated Recurrent Units (GRUs)** are a simpler variant of LSTMs, introduced to reduce computational complexity while often achieving comparable performance. They streamline the LSTM architecture by combining some of the gates and merging the cell state with the hidden state.

**Key Components of a GRU Cell:**
*   **Update Gate (`z_t`):** This gate combines the functionality of the forget and input gates from an LSTM. It decides how much of the past information (from the previous hidden state) should be carried forward to the current hidden state.
*   **Reset Gate (`r_t`):** This gate decides how much of the previous hidden state to "forget" or ignore when computing the new candidate hidden state.
*   **Candidate Hidden State (`\tilde{h}_t`):** This is a new potential hidden state, computed using the current input and the previous hidden state (after applying the reset gate).

The final hidden state (`h_t`) is then a linear combination of the previous hidden state and the candidate hidden state, weighted by the update gate.

## 5. Comparing RNNs, LSTMs, and GRUs

| Feature                 | Standard RNNs                               | LSTMs                                                              | GRUs                                                               |
| :---------------------- | :------------------------------------------ | :----------------------------------------------------------------- | :----------------------------------------------------------------- |
| **Memory Mechanism**    | Simple hidden state                         | Complex cell state with three gates                                | Combined hidden/cell state with two gates                          |
| **Long-Term Dependencies** | Struggle due to vanishing/exploding gradients | Excellent at capturing                                             | Good at capturing (often comparable to LSTMs)                      |
| **Complexity**          | Simplest                                    | More complex, more parameters                                      | Simpler than LSTMs, fewer parameters                               |
| **Training Speed**      | Fastest (but less effective)                | Slower than GRUs due to more computations                          | Faster than LSTMs                                                  |
| **Performance**         | Limited for long sequences                  | State-of-the-art for many sequence tasks                           | Often comparable to LSTMs, good alternative                        |
| **Use Case**            | Short sequences, simple patterns            | Complex sequence tasks, long dependencies (e.g., machine translation) | Good general-purpose choice, especially with limited data/resources |

The choice between LSTMs and GRUs often depends on the specific task, dataset size, and available computational resources. LSTMs are generally more powerful for very long sequences, while GRUs offer a good balance of performance and efficiency.

## 6. Advanced Concepts

*   **Bidirectional RNNs/LSTMs/GRUs:** Process the input sequence in both forward and backward directions, allowing the model to capture context from both past and future elements in the sequence. This is particularly useful in tasks like machine translation or sentiment analysis.
*   **Stacked RNNs/LSTMs/GRUs:** Multiple layers of recurrent units are stacked on top of each other, allowing the network to learn more abstract and complex representations of the sequential data. The output of one recurrent layer serves as the input to the next.

---
## Recommended Reading

-   **Neural Networks and Deep Learning** — *Michael Nielsen* (FREE online)
-   **Deep Learning** — *Goodfellow et al.* (Chapter 10: Sequence Modeling)
-   **Dive Into Deep Learning (D2L)**: RNN, LSTM, GRU chapters
-   **Understanding LSTM Networks** — *Chris Olah* (blog, classic explainer)
-   **Natural Language Processing with PyTorch** — *Rao & McMahan* (RNN chapters)
