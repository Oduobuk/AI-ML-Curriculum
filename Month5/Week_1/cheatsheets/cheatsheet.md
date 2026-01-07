Cheatsheet: Fundamentals of Reinforcement Learning

 1. The Reinforcement Learning Problem
RL is about an agent learning to make optimal decisions by interacting with an environment.
- The agent takes an action.
- The environment returns a new state and a reward.
- The agent's goal is to learn a policy (a strategy) that maximizes its cumulative reward over time.

---

 2. Markov Decision Process (MDP)
The mathematical framework for RL problems. An MDP consists of:
- S: A set of all possible States.
- A: A set of all possible Actions.
- P: The Transition Probability Function. `P(s' | s, a)` is the probability of transitioning to state `s'` after taking action `a` in state `s`.
- R: The Reward Function. `R(s, a, s')` is the immediate reward for transitioning from `s` to `s'` via action `a`.
- γ (Gamma): The Discount Factor (0 ≤ γ < 1). It determines the importance of future rewards. A value of 0 makes the agent short-sighted, while a value near 1 makes it strive for long-term rewards.

The Markov Property: The future is independent of the past, given the present. The current state `s` contains all necessary information for the agent to make a decision.

---

 3. Policies and Value Functions
- Policy (π): The agent's strategy or "brain". It maps states to actions. `π(a|s)` is the probability of taking action `a` in state `s`. The goal of RL is to find the optimal policy, `π`.
- State-Value Function (V(s)): The expected total reward starting from state `s` and then following a policy `π`. "How good is it to be in this state?"
- Action-Value Function (Q(s, a)): The expected total reward starting from state `s`, taking action `a`, and then following a policy `π`. "How good is it to take this action in this state?"

---

 4. Value-Based Methods: Q-Learning
These methods learn a value function (typically the Q-function) and then derive a policy from it.

 Q-Learning
- Model-Free: It doesn't need to know the transition probabilities (P) or reward function (R).
- Off-Policy: It can learn the optimal policy even while following a different, exploratory policy.
- Q-Table: For discrete state/action spaces, this is a simple lookup table storing the Q-value for every `(state, action)` pair.
- The Bellman Update Equation: The core of Q-learning. It iteratively updates the Q-table.
  `Q(s, a) ← Q(s, a) + α  [r + γ  max_a' Q(s', a') - Q(s, a)]`
    - `α` (alpha) is the learning rate.

 Exploration vs. Exploitation
- Exploration: Trying new actions to discover potentially better rewards.
- Exploitation: Choosing the action known to give the best reward.
- Epsilon-Greedy Strategy: A common way to balance this. With probability `ε` (epsilon), choose a random action (explore). Otherwise, choose the best-known action (exploit). `ε` often decays over time.

---

 5. Policy-Based Methods: Policy Gradients
These methods directly learn the parameters of the policy `π(a|s; θ)` without needing to learn a value function first.

- Core Idea: Adjust the policy's parameters `θ` in the direction that makes actions leading to high rewards more likely.
- REINFORCE Algorithm: A foundational policy gradient algorithm. It uses the return from an entire episode to update the policy.
- Advantage: Policy gradient methods can handle continuous action spaces and learn stochastic policies naturally.
- Disadvantage: Often have high variance in gradient estimates, which can make training unstable. This is often addressed by using a baseline or Actor-Critic methods.