Exercises: Fundamentals of Reinforcement Learning

 Exercise 1: Frame a Problem as an MDP

Consider the simple game of Tic-Tac-Toe. Frame this game as a Markov Decision Process (MDP) by defining its components.

1.  States (S): How would you represent a state in Tic-Tac-Toe? What is the total number of possible states? (A rough estimate is fine).
2.  Actions (A): What are the possible actions an agent can take at each turn?
3.  Reward Function (R): Define a simple reward function. What reward would you give for a win, a loss, a draw, and for a move that doesn't end the game?
4.  Discount Factor (γ): Would you set the discount factor closer to 0 or 1 for this game? Explain your reasoning.

 Exercise 2: The Bellman Update in Q-Learning

Imagine an agent is in State A. The agent can move `left` or `right`.
- The Q-table currently has the following values:
    - `Q(A, left) = 5`
    - `Q(A, right) = 6`
- The agent decides to take the action `right`.
- After moving `right`, it lands in State B and receives an immediate reward of -1.
- From State B, the possible actions are `up` and `down`.
- The Q-table values for State B are:
    - `Q(B, up) = 10`
    - `Q(B, down) = 8`

You are given the following hyperparameters:
- Learning Rate (α) = 0.1
- Discount Factor (γ) = 0.9

Task: Calculate the new Q-value for `Q(A, right)` using the Bellman update equation:
`Q(s, a) ← Q(s, a) + α  [r + γ  max_a' Q(s', a') - Q(s, a)]`

Show your work step-by-step.

 Exercise 3: Exploration vs. Exploitation

An agent is using an epsilon-greedy strategy with `epsilon = 0.2`. The agent is in a state where the Q-values for the possible actions are:
- `Q(s, action1) = 15`
- `Q(s, action2) = 25`
- `Q(s, action3) = 10`

1.  Exploitation: If the agent decides to exploit, which action will it choose?
2.  Exploration: If the agent decides to explore, what happens?
3.  Probability: What is the probability that the agent will choose `action2` on this turn?

 Exercise 4: Value-Based vs. Policy-Based Methods

1.  What is the fundamental difference in what Q-Learning (a value-based method) and REINFORCE (a policy-based method) are trying to learn?
2.  Which of these two methods would be more suitable for a problem where the actions are continuous (e.g., controlling a robot arm where the action is the precise angle of the joint)? Why?