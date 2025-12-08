 Month 3, Week 4: Conceptual Exercises

These questions are designed to test your understanding of the core concepts behind neural networks and the backpropagation algorithm.



 Question 1: The Role of Non-Linearity

The lecture emphasizes that an activation function `g(z)` must be non-linear. What would happen if you built a deep, multi-layer neural network but used a purely linear activation function, such as `g(z) = z`? Would the network still be able to learn complex patterns? Why or why not?



 Question 2: The Bias Term

What is the purpose of the bias term in a neuron's calculation (`weighted_sum + bias`)? What would be the limitation of a neuron that had weights but no bias term? (Hint: Think about the weighted sum `z` and what value it must pass for the neuron to activate).



 Question 3: Weight Initialization

In the code examples, weights are initialized to small, random numbers (e.g., `np.random.normal(...)`).
a) Why not initialize all weights to zero? What problem would this cause during training?
b) Why not initialize them to large random numbers?



 Question 4: The Learning Rate

In the gradient descent update rule (`W = W - α  dW`), `α` (alpha) is the learning rate.
a) What would happen if you set the learning rate to be very large?
b) What would happen if you set it to be very small?
c) Does the ideal learning rate stay the same throughout training, or might you want to change it?



 Question 5: Backpropagation Intuition

The backpropagation algorithm calculates the gradient of the cost function with respect to a weight `W_ij` (connecting neuron `j` in layer `L-1` to neuron `i` in layer `L`).

The final update depends on three main factors:
1.  The activation of the input neuron `a_j^[L-1]`.
2.  The error of the output neuron `δ_i^[L]`.
3.  The learning rate `α`.

Explain in your own words why the weight update should be proportional to the input activation `a_j^[L-1]`. (Hint: Think about the phrase "neurons that fire together, wire together").



 Question 6: Overfitting

Imagine you train your MNIST classifier for a very long time with a very powerful (many layers, many neurons) network. You observe that your accuracy on the training data reaches 99.99%, but your accuracy on the test data (data the network has never seen) is only 85% and seems to be getting worse.

a) What is this phenomenon called?
b) Why does it happen?
c) What are some high-level strategies you could use to combat this?
