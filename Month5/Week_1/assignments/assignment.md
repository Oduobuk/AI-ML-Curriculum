Assignment: Q-Learning with FrozenLake

 Objective

The goal of this assignment is to implement the Q-Learning algorithm from scratch to solve the "FrozenLake-v1" environment from the OpenAI Gym library. This will be your first practical implementation of a reinforcement learning agent.

You will need to create and update a Q-table while balancing exploration and exploitation to teach an agent to navigate a slippery, frozen lake to find a goal.

 The FrozenLake Environment

"FrozenLake-v1" is a classic grid-world environment.
- Goal: Navigate from a starting state 'S' to a goal state 'G' across a frozen lake.
- States: The grid is 4x4, so there are 16 states (0-15).
- Actions: There are 4 discrete actions: 0 (left), 1 (down), 2 (right), 3 (up).
- Rewards: The agent receives a reward of +1 for reaching the goal 'G' and 0 for all other states.
- Challenge: The ice is "slippery". The agent does not always move in the intended direction. There are also "holes" ('H') in the ice; falling into a hole ends the episode.

 Requirements

 1. Setup
- Make sure you have `gym` installed (`pip install gym`).
- Create the environment using `gym.make("FrozenLake-v1")`.

 2. Q-Learning Implementation
You must implement the Q-Learning algorithm from scratch. Do not use a pre-built library for the algorithm itself.

Your implementation should include:
- Q-Table Initialization: Initialize a Q-table with zeros. The size should be `(number_of_states, number_of_actions)`.
- Hyperparameters: Define the following hyperparameters:
    - `learning_rate` (alpha)
    - `discount_factor` (gamma)
    - `num_episodes` (total episodes to train for)
    - `max_steps_per_episode`
    - `exploration_rate` (epsilon) and its `decay_rate`.
- The Main Training Loop:
    - Loop for `num_episodes`.
    - In each episode, reset the environment to get the initial state.
    - Loop for `max_steps_per_episode`:
        - Exploration-Exploitation: Implement the epsilon-greedy strategy to choose an action.
        - Take Action: Perform the chosen action in the environment using `env.step()`.
        - Update Q-Table: Update the Q-value for the `(state, action)` pair using the Bellman equation: `Q(s, a) ← Q(s, a) + α  [r + γ  max_a' Q(s', a') - Q(s, a)]`.
        - Update the current state.
        - Break the inner loop if the episode is `done`.
    - Epsilon Decay: After each episode, decay the `exploration_rate`. A common decay formula is `epsilon = min_epsilon + (max_epsilon - min_epsilon)  exp(-decay_rate  episode)`.

 3. Demonstrate the Learned Policy
- After the training loop is finished, your Q-table should contain the learned values.
- Run a few (e.g., 3-5) episodes where the agent only exploits (i.e., `epsilon = 0`).
- Print the results of these demonstration episodes to show whether your agent has learned to reliably reach the goal.

 Submission

Submit a single Python script (`.py`) or a Jupyter Notebook (`.ipynb`) file containing your complete, executable code. The code should perform all the steps from environment setup to training and final demonstration. Include comments to explain the different parts of your Q-learning implementation.

 Grading Criteria
- Q-Learning Implementation (60%): Is the Q-table initialized and updated correctly according to the Bellman equation? Is the training loop structured properly?
- Exploration-Exploitation (20%): Is the epsilon-greedy strategy correctly implemented? Is the exploration rate decayed over time?
- Demonstration and Results (20%): Does the code demonstrate the final learned policy? Does the agent show clear evidence of learning by successfully reaching the goal more often than not?