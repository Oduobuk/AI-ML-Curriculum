# Week 1 Cheatsheet: AI Fundamentals

## The Three Stages of AI

| Stage                          | AKA        | Description                                                                 | Current Status                               |
| ------------------------------ | ---------- | --------------------------------------------------------------------------- | -------------------------------------------- |
| **Artificial Narrow Intelligence** | Weak AI    | AI that is designed and trained for one specific task.                      | **Achieved.** All current AI is Narrow AI.   |
| **Artificial General Intelligence**| Strong AI  | A machine with the ability to learn and understand any task a human can.    | **Not Achieved.** This is a research goal.   |
| **Artificial Super Intelligence**  | ASI        | An intellect that is much smarter than the best human brains in every field. | **Hypothetical.** The realm of science fiction. |

---

## The Three Types of Machine Learning

| Type                  | How it Learns                                       | Data Requirement          | Goal                                               | Analogy                                       |
| --------------------- | --------------------------------------------------- | ------------------------- | -------------------------------------------------- | --------------------------------------------- |
| **Supervised Learning** | Learns from examples with correct answers.          | **Labeled Data** (Input + Output) | Predict an output for new, unseen data.            | A student learning with a teacher and flashcards. |
| **Unsupervised Learning** | Finds patterns and structures on its own.           | **Unlabeled Data** (Input only) | Discover hidden groupings or anomalies in the data. | A detective finding patterns among case files.    |
| **Reinforcement Learning**| Learns through trial and error with rewards/penalties. | **No Pre-defined Data**   | Learn the best sequence of actions to take.        | A dog learning a new trick with treats as rewards. |

---

## Core Components of a Neural Network

*   **Neuron:** A node that holds a numerical value called an **activation** (usually between 0 and 1).
*   **Layers:** Neurons are organized into layers.
    *   **Input Layer:** Receives the initial data. The number of neurons often matches the number of features in the data.
    *   **Hidden Layers:** Intermediate layers where most of the "learning" happens. They detect patterns in the data.
    *   **Output Layer:** Produces the final result or prediction.
*   **Weights:** Numbers that define the strength of the connection between two neurons. The network learns by adjusting these weights.
*   **Bias:** A number added to the weighted sum of inputs, which helps shift the activation function. It determines how easily a neuron gets activated.
*   **Activation Function (e.g., Sigmoid, ReLU):** A function that "squishes" the output of a neuron into a specific range (like 0 to 1), adding non-linearity to the network.

---

## Why Python for AI?

*   **Simple Syntax:** Easy to read and write.
*   **Massive Libraries:**
    *   **NumPy:** For numerical operations.
    *   **Pandas:** For data manipulation.
    *   **Scikit-learn:** For classic machine learning algorithms.
    *   **TensorFlow & PyTorch:** For deep learning.
*   **Large Community:** Easy to find help and resources.
