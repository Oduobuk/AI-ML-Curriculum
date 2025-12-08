 Month 3, Week 4: Coding Assignment - Extending the Neural Network

 Objective

In this assignment, you will take the `BackPropagationNetwork` class from the lecture and code examples and extend its functionality. This will solidify your understanding of how different components of a neural network work together.

You will be working with the `neural_network_from_scratch.py` file provided in the `02_Code_Examples` folder. Make a copy of this file to work on for your assignment.



 Task 1: Implement the `train` Method (Core Task)

The provided code in the lecture notes only has a `run` method (forward propagation). Your first and most important task is to implement the `train` method. This will involve coding the backpropagation algorithm.

Steps:

1.  Define the `train` method signature:
    It should accept `self`, `input_data`, `labels`, `learning_rate`, and `epochs`.

2.  Create the training loop:
    The method should loop for the specified number of `epochs`.

3.  Perform a Forward Pass:
    Inside the loop, the first step is to call `self.run(input_data)` to execute a forward pass. This will compute the network's predictions and, crucially, cache the inputs and outputs of each layer (`self._layer_input` and `self._layer_output`), which you'll need for the backward pass.

4.  Implement the Backward Pass (Backpropagation):
    This is the most complex part. You need to calculate the error "delta" for each layer, starting from the output and moving backward.

       Delta for the Output Layer: The error at the output layer is the difference between the network's predictions and the true labels.
        `delta_output = network_output - true_labels`

       Delta for Hidden Layers: To calculate the delta for a hidden layer, you need to "propagate" the error from the layer in front of it. The formula is:
        `delta_hidden = (weights_from_hidden_to_next_layer.T @ delta_next_layer)  derivative_of_hidden_activation`

           You will need to loop backward from the last hidden layer to the first.
           Remember to exclude the bias weight during error propagation.
           The `derivative_of_hidden_activation` is the derivative of the activation function (e.g., sigmoid or ReLU) applied to the output of that hidden layer.

5.  Update the Weights:
    After calculating the deltas for all layers, you can now calculate the gradient for each weight matrix and update the weights.

       The gradient for a layer's weights is the dot product of that layer's delta and its input, averaged over all samples:
        `gradient = delta_layer @ input_to_layer.T / number_of_samples`

       Update the weights using the gradient descent rule:
        `self.weights[i] -= learning_rate  gradient`

6.  Add Progress Indicators:
    Inside your training loop, print the loss and accuracy every 100 epochs so you can see if your network is learning.



 Task 2: Add a New Activation Function

The `sigmoid` function is classic but can lead to issues like vanishing gradients. A more modern and common choice for hidden layers is the Rectified Linear Unit (ReLU).

Steps:

1.  Implement the `relu` function:
    Create a new function `relu(x, derivative=False)`.
       The function itself is `np.maximum(0, x)`.
       Its derivative is `1` for `x > 0` and `0` otherwise. A simple way to implement this is `np.where(x > 0, 1, 0)`.

2.  Modify the `__init__` method:
    Allow the `BackPropagationNetwork` class to accept a list of activation functions for its hidden layers. If none are provided, it should default to using `sigmoid`.

3.  Modify the `run` and `train` methods:
    Update the forward and backward passes to use the specified activation function (and its derivative) for each hidden layer. The output layer should still use `softmax` (or `sigmoid` for binary classification).



 Task 3: (Optional Challenge) Implement Cross-Entropy Loss

Our basic implementation uses Mean Squared Error (MSE) as the implicit loss function, which leads to the simple `(prediction - label)` error delta for the output layer when using a linear output. For classification tasks, Categorical Cross-Entropy is a much more effective loss function, especially when paired with a `softmax` output layer.

Steps:

1.  Implement the `softmax` function:
    This function takes the raw outputs (logits) of the final layer and converts them into probabilities that sum to 1.
    `softmax(x) = np.exp(x) / np.sum(np.exp(x))`
    Hint: For numerical stability, subtract the max value from `x` before exponentiating: `np.exp(x - np.max(x))`.

2.  Modify the `run` method:
    Ensure the final layer of your network uses the `softmax` function.

3.  Simplify the Output Delta:
    The beauty of using softmax with cross-entropy is that the error delta for the output layer remains incredibly simple:
    `delta_output = network_output_after_softmax - true_labels`
    You don't need to calculate the derivative of softmax separately; it elegantly simplifies to this form.

4.  Calculate and Print the Loss:
    Inside your training loop, calculate the categorical cross-entropy loss to monitor progress:
    `loss = -np.sum(true_labels  np.log(predictions + 1e-9)) / number_of_samples`
    (The `1e-9` is added to prevent `log(0)` errors).

Good luck!
