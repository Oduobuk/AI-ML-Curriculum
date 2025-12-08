# Month 5, Week 1: Reinforcement Learning Conceptual Exercises

## Instructions
These exercises are designed to test your understanding of the core concepts of Reinforcement Learning. Write down your answers and be prepared to discuss them.

## Exercises

1.  **The Discount Factor (γ):**
    *   What is the role of the discount factor γ in Reinforcement Learning?
    *   What happens if γ = 0? What kind of agent would this create?
    *   What happens if γ = 1? What are the potential challenges with this setting?

2.  **Markov Decision Process (MDP):**
    *   What is the Markov Property? Why is it important for RL?
    *   Think of a real-world scenario that can be modeled as an MDP. Identify the states, actions, and rewards. Does it perfectly satisfy the Markov Property? If not, why?

3.  **On-Policy vs. Off-Policy Learning:**
    *   What is the fundamental difference between on-policy and off-policy learning?
    *   SARSA is an on-policy algorithm, while Q-Learning is an off-policy algorithm. Explain how their update rules reflect this difference.
    *   Why might you choose an off-policy algorithm over an on-policy one? What are the potential advantages?

4.  **Model-Based vs. Model-Free RL:**
    *   What is a "model" in the context of Reinforcement Learning?
    *   Give an example of a model-based RL algorithm and a model-free RL algorithm.
    *   What are the trade-offs between model-based and model-free approaches? In what situations would one be preferable to the other?

5.  **Designing a Reward Signal:**
    *   Imagine you are designing an RL agent to learn to play chess. How would you design the reward signal?
    *   Consider different reward schemes:
        *   A sparse reward: +1 for winning, -1 for losing, 0 for all other moves.
        *   A dense reward: small rewards for capturing pieces, achieving a good board position, etc.
    *   What are the pros and cons of each approach? How might the choice of reward signal affect the agent's learning process and final performance?

6.  **Value Function Estimation:**
    *   Both Monte Carlo and Temporal-Difference (TD) learning are used to estimate value functions from experience.
    *   What is the key difference in how they update their value estimates?
    *   What is bootstrapping? Which of these methods uses bootstrapping?
    *   Discuss the bias-variance trade-off for Monte Carlo and TD methods.
