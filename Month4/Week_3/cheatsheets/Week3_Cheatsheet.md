# Month 4, Week 3 Cheatsheet: RNNs, LSTMs, and GRUs

## 1. Recurrent Neural Networks (RNNs)

*   **Purpose:** Designed to process **sequential data** where order matters (e.g., text, time series, speech).
*   **Key Idea:** Maintain an internal **hidden state** (`h_t`) that acts as a memory of previous inputs.
*   **Recurrent Connection:** Output from hidden layer at `t-1` is fed as input to hidden layer at `t`.
*   **Unrolling:** Visualizing the RNN as a deep feedforward network over time steps.
*   **Limitations:**
    *   **Vanishing/Exploding Gradients:** Struggle to learn **long-term dependencies** due to gradients shrinking/growing too much during Backpropagation Through Time (BPTT).
    *   Cannot effectively "remember" information over long sequences.

---

## 2. Long Short-Term Memory (LSTM) Networks

*   **Purpose:** Overcome vanishing gradient problem in RNNs; excel at learning **long-term dependencies**.
*   **Core Component:** **Cell State (`C_t`)** – a horizontal line running through the LSTM, acting as the network's memory.
*   **Gates:** Control the flow of information into and out of the cell state. Each gate is a sigmoid layer (`σ`) followed by a pointwise multiplication.
    *   **Forget Gate (`f_t`):** Decides what information to discard from `C_{t-1}`.
        *   `f_t = σ(W_f · [h_{t-1}, x_t] + b_f)`
    *   **Input Gate (`i_t`):** Decides what new information to store in `C_t`.
        *   `i_t = σ(W_i · [h_{t-1}, x_t] + b_i)`
        *   `\tilde{C}_t = tanh(W_C · [h_{t-1}, x_t] + b_C)` (Candidate for new memory)
    *   **Update Cell State:** `C_t = f_t * C_{t-1} + i_t * \tilde{C}_t`
    *   **Output Gate (`o_t`):** Decides what part of `C_t` to output as `h_t`.
        *   `o_t = σ(W_o · [h_{t-1}, x_t] + b_o)`
        *   `h_t = o_t * tanh(C_t)`

---

## 3. Gated Recurrent Units (GRUs)

*   **Purpose:** A simpler, more computationally efficient variant of LSTMs, often with comparable performance.
*   **Key Idea:** Combines the forget and input gates into a single **Update Gate**, and merges the cell state with the hidden state.
*   **Gates:**
    *   **Update Gate (`z_t`):** Controls how much of the past information (`h_{t-1}`) to carry forward and how much new information (`\tilde{h}_t`) to add.
        *   `z_t = σ(W_z · [h_{t-1}, x_t] + b_z)`
    *   **Reset Gate (`r_t`):** Controls how much of the past information (`h_{t-1}`) to forget when computing the new candidate hidden state.
        *   `r_t = σ(W_r · [h_{t-1}, x_t] + b_r)`
    *   **Candidate Hidden State (`\tilde{h}_t`):**
        *   `\tilde{h}_t = tanh(W_h · [r_t * h_{t-1}, x_t] + b_h)`
    *   **Update Hidden State:** `h_t = (1 - z_t) * h_{t-1} + z_t * \tilde{h}_t`

---

## 4. Comparison

| Feature                 | RNNs                                      | LSTMs                                         | GRUs                                          |
| :---------------------- | :---------------------------------------- | :-------------------------------------------- | :-------------------------------------------- |
| **Complexity**          | Low                                       | High                                          | Medium                                        |
| **Parameters**          | Fewest                                    | Most                                          | Fewer than LSTMs                              |
| **Long-Term Memory**    | Poor                                      | Excellent                                     | Very Good                                     |
| **Training Speed**      | Fast (but often ineffective)              | Slower                                        | Faster than LSTMs                             |
| **Performance**         | Limited for complex sequences             | High, widely used for state-of-the-art results | Often comparable to LSTMs, good for efficiency |
| **Gates**               | None (simple recurrence)                  | Forget, Input, Output                         | Update, Reset                                 |
| **Cell State**          | No explicit cell state                    | Yes                                           | No (hidden state acts as memory)              |

---

## 5. Advanced Concepts

*   **Bidirectional RNNs/LSTMs/GRUs:** Process sequence in both forward and backward directions to capture context from both past and future.
*   **Stacked RNNs/LSTMs/GRUs:** Multiple layers of recurrent units, where the output of one layer feeds into the next, allowing for learning of more abstract representations.
