Policy Gradients

Policy Gradients are a class of reinforcement learning algorithms that directly optimize an agent's policy—its strategy for choosing actions—to maximize cumulative rewards. Unlike value-based methods (like Q-learning) that learn the value of state-action pairs and then derive a policy, policy gradient methods directly learn the parameters of the policy itself.

 The Core Idea

The fundamental principle behind policy gradient methods is to adjust the policy's parameters in the direction that increases the expected total reward. This is achieved by calculating the gradient of the expected return with respect to the policy parameters and then updating the parameters using gradient ascent.

The general process involves these steps:

1.  Rollout: The agent interacts with the environment using its current policy, collecting a sequence of states, actions, and rewards.
2.  Compute Returns: For each step in the trajectory, the cumulative future reward (the return) is calculated.
3.  Estimate the Gradient: The gradient of the objective function (the expected return) is estimated using the collected trajectories.
4.  Update the Policy: The policy's parameters are updated in the direction of the calculated gradient to make actions that led to higher returns more likely in the future.

 The Policy Gradient Theorem

The Policy Gradient Theorem provides a way to compute the gradient of the expected return without needing to model the environment. The theorem states that the gradient of the expected return is proportional to the expectation of the product of the return and the gradient of the logarithm of the policy probability.

∇J(θ) = E [ R(τ)  ∇log π(a|s; θ) ]

Where:

    `J(θ)` is the expected return of the policy.
    `θ` represents the parameters of the policy `π`.
    `R(τ)` is the return of a trajectory `τ`.
    `∇log π(a|s; θ)` is the gradient of the log-probability of taking action `a` in state `s`.

 REINFORCE Algorithm

The REINFORCE algorithm, also known as Monte Carlo Policy Gradient, is a direct implementation of the Policy Gradient Theorem. It learns from complete episodes of experience to update the policy.

A significant drawback of the basic REINFORCE algorithm is the high variance in the gradient estimates, which can make training unstable.

 Reducing Variance: Baselines and the Advantage Function

To address the high variance of REINFORCE, a baseline is often introduced into the policy gradient update. The baseline is a value that is subtracted from the return and is a function of the state. A common choice for the baseline is the state-value function, V(s).

The introduction of a baseline leads to the concept of the advantage function, A(s, a), which is defined as:

A(s, a) = Q(s, a) - V(s)

By using the advantage function in the policy gradient update, the algorithm focuses on whether an action is better or worse than the average, rather than just whether the return was high or low.

 Actor-Critic Methods

Actor-Critic methods are a family of policy gradient algorithms that explicitly learn both a policy (the actor) and a value function (the critic).

    The Actor: The actor is the policy, which decides which action to take in a given state.
    The Critic: The critic evaluates the actions taken by the actor by learning a value function.

The critic's evaluation provides a low-variance estimate of the policy gradient for the actor to update its parameters.