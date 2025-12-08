# Month 5, Week 1: Reinforcement Learning Assignment

## Instructions
Answer the following questions to the best of your ability. For the programming task, you can use any language you are comfortable with, but Python is recommended.

## Theoretical Questions

1.  **Core Components:** In your own words, describe the relationship between an agent, an environment, a state, an action, and a reward in the context of Reinforcement Learning.
2.  **Policy and Value Function:** What is the difference between a policy and a value function? How do they relate to each other?
3.  **Bellman Equation:** Explain the significance of the Bellman equation in Reinforcement Learning. Write down the Bellman equation for the state-value function V(s).
4.  **Exploration vs. Exploitation:** Why is the exploration-exploitation trade-off a central challenge in RL? Describe a simple strategy to balance exploration and exploitation.
5.  **Learning Methods:** Briefly compare and contrast Dynamic Programming, Monte Carlo methods, and Temporal-Difference learning. When would you choose one over the others?

## Programming Task

**Objective:** Implement a simple text-based game environment and a Q-learning agent to solve it.

**Environment:**
*   A 1D world represented by a 10-cell array: `['-', '-', '-', '-', 'G', '-', '-', '-', 'H', '-']`
*   `'-'` is a safe cell.
*   `'G'` is the goal.
*   `'H'` is a hole.
*   The agent starts at index 0.
*   **Actions:** The agent can move 'left' or 'right'.
*   **Rewards:**
    *   +10 for reaching the Goal ('G').
    *   -10 for falling into a Hole ('H').
    *   -1 for any other move.
*   **Episode End:** The episode ends when the agent reaches 'G' or 'H'.

**Task:**
1.  Implement the environment described above. It should have a `step(action)` method that returns the `(next_state, reward, done)` tuple.
2.  Implement a Q-learning agent that learns a policy to navigate this environment.
3.  Train the agent for a sufficient number of episodes.
4.  Print the final learned Q-table.

**Submission:**
*   A single markdown file with your answers to the theoretical questions.
*   A single script file (e.g., `q_learning.py`) with your implementation of the programming task.
