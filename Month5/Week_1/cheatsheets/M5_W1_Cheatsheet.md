# Month 5, Week 1: Reinforcement Learning Cheatsheet

## Core Concepts

| Term | Description |
| :--- | :--- |
| **Agent** | The learner or decision-maker that interacts with the environment. |
| **Environment** | The external system that the agent interacts with, providing states and rewards. |
| **State (s)** | A representation of the environment at a specific point in time. |
| **Action (a)** | A decision made by the agent to interact with the environment. |
| **Reward (r)** | The immediate feedback signal the agent receives from the environment after performing an action. |
| **Policy (π)** | The agent's strategy for choosing actions in different states. Can be deterministic or stochastic. |
| **Value Function (V, Q)** | A prediction of the expected future reward. Used to evaluate the goodness of states or state-action pairs. |
| **Episode** | A sequence of states, actions, and rewards from a starting state to a terminal state. |

## Key Equations

**Bellman Equation for State-Value Function V(s):**
```
V(s) = E[R_{t+1} + γ * V(S_{t+1}) | S_t = s]
```
This equation expresses the value of a state as the expected immediate reward plus the discounted value of the next state.

**Bellman Equation for Action-Value Function Q(s, a):**
```
Q(s, a) = E[R_{t+1} + γ * Q(S_{t+1}, A_{t+1}) | S_t = s, A_t = a]
```
This equation expresses the value of taking an action in a state as the expected immediate reward plus the discounted value of the next state-action pair.

## Algorithms

| Algorithm | Learning Method | Model-Based/Free | Description |
| :--- | :--- | :--- | :--- |
| **Value Iteration** | Dynamic Programming | Model-Based | Iteratively updates the value function until it converges to the optimal value function. |
| **Policy Iteration** | Dynamic Programming | Model-Based | Alternates between policy evaluation (calculating the value function for the current policy) and policy improvement (improving the policy based on the current value function). |
| **Monte Carlo (MC)** | Monte Carlo | Model-Free | Learns from complete episodes of experience. Updates value estimates after an episode is finished. |
| **Q-Learning** | Temporal-Difference | Model-Free | An off-policy TD algorithm that learns the optimal action-value function, Q*(s, a), directly. |
| **SARSA** | Temporal-Difference | Model-Free | An on-policy TD algorithm that learns the action-value function for the current policy. |

## Exploration vs. Exploitation

*   **Exploration:** Trying out new actions to discover their potential rewards.
*   **Exploitation:** Choosing the action with the currently estimated best value.

**Common Strategies:**
*   **ε-greedy (Epsilon-greedy):** With probability ε, choose a random action (explore). With probability 1-ε, choose the best-known action (exploit).
*   **Upper Confidence Bound (UCB):** Choose actions based on both their estimated value and the uncertainty of that estimate.
*   **Softmax (Boltzmann) Exploration:** Choose actions based on a probability distribution derived from their value estimates.
