Cheatsheet: Recurrent Neural Networks (RNNs)

 The Problem with Sequential Data
Traditional neural networks process inputs independently. RNNs are designed for sequential data (e.g., text, time series, speech) where context and order matter. They do this by using a hidden state (a form of memory) that is passed from one timestep to the next.

---

 Core RNN Architectures

| Layer | Key Idea | Pros | Cons | Keras Layer |
| :--- | :--- | :--- | :--- | :--- |
| Simple RNN | A basic loop. The output at time `t` is a function of the input at `t` and the hidden state from `t-1`. | Simple, computationally cheap. | Suffers from vanishing/exploding gradients. Cannot learn long-term dependencies. | `SimpleRNN` |
| LSTM | A more complex unit with a dedicated cell state (long-term memory) and three "gates" to control information flow. | Solves the vanishing gradient problem. Can learn very long-term dependencies. | More parameters, computationally more expensive than GRU. | `LSTM` |
| GRU | A simplified version of LSTM. Combines the forget and input gates into a single update gate. | Also solves the vanishing gradient problem. Fewer parameters and faster to train than LSTM, with often comparable performance. | Slightly less expressive than LSTM on some tasks. | `GRU` |

---

 The Vanishing / Exploding Gradient Problem
- Why it happens: In a simple RNN, gradients are repeatedly multiplied by the same weight matrix as they are backpropagated through time.
- Vanishing Gradient: If weights are small (<1), the gradient signal shrinks exponentially, preventing the network from learning long-range dependencies. This is the more common problem.
- Exploding Gradient: If weights are large (>1), the gradient signal grows exponentially, leading to unstable training.
- Solutions:
    1.  Gradient Clipping: A simple fix for exploding gradients. If a gradient's norm exceeds a threshold, it is scaled down.
    2.  Gated Architectures (LSTMs/GRUs): The primary solution for the vanishing gradient problem. Their gating mechanisms allow gradients to flow more easily through time.

---

 LSTM & GRU Gates

 LSTM (Long Short-Term Memory)
An LSTM cell has three gates and a cell state:
1.  Forget Gate (Sigmoid): Decides what information to throw away from the cell state.
2.  Input Gate (Sigmoid & Tanh): Decides what new information to store in the cell state.
3.  Output Gate (Sigmoid & Tanh): Decides what to output from the cell state to the hidden state.
- Cell State: The "conveyor belt" of memory. Information can flow along it with minimal changes, allowing long-term dependencies to be preserved.

 GRU (Gated Recurrent Unit)
A GRU cell has two gates and no separate cell state:
1.  Update Gate (Sigmoid): Acts like a combination of the forget and input gates. It decides how much of the past hidden state to keep and how much of the new candidate hidden state to use.
2.  Reset Gate (Sigmoid): Decides how much of the past hidden state to forget when proposing a new candidate hidden state.