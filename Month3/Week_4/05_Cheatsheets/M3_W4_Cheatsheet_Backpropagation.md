 Cheatsheet: Backpropagation & Neural Networks

This cheatsheet summarizes the key formulas and concepts for a simple feedforward neural network.



 1. Core Components

- Neuron Activation (`a`): A value between 0 and 1 representing how "active" a neuron is.
- Weights (`W`): A matrix of parameters that scales the activations from the previous layer. `W_ij` is the weight from the j-th neuron in the previous layer to the i-th neuron in the current layer.
- Bias (`b`): A vector of parameters that provides an offset, allowing the activation function to shift.
- Weighted Sum (`z`): The linear combination of inputs, weights, and bias before the activation function. `z = W  a_prev + b`.
- Activation Function (`g(z)`): A non-linear function that "squishes" the weighted sum `z` into the neuron's output activation `a`. `a = g(z)`.



 2. Key Activation Functions & Their Derivatives

| Function | Formula `g(z)` | Derivative `g'(z)` |
| : | : | : |
| Sigmoid | `1 / (1 + exp(-z))` | `g(z)  (1 - g(z))` or `a  (1 - a)` |
| ReLU | `max(0, z)` | `1` if `z > 0`, `0` otherwise |
| Softmax | `exp(z_i) / sum(exp(z_j))` | (Used for output layer, derivative is complex but simplifies in loss function) |



 3. Forward Propagation

For a single layer `L`:

1.  Calculate Weighted Sum:
    `z^[L] = W^[L] @ a^[L-1] + b^[L]`

2.  Calculate Activation (Output of the layer):
    `a^[L] = g(z^[L])`

(Repeat for all layers from input to output)



 4. Cost Function (Categorical Cross-Entropy)

For a single sample: `C = - sum(y_i  log(a_i))`
For all `m` samples: `C = -(1/m)  sum(Y  log(A))`

- `y_i` is the true label (1 for the correct class, 0 otherwise).
- `a_i` is the predicted probability for that class.
- `Y` and `A` are matrices for all samples.



 5. Backpropagation (The Four Core Equations)

Let `δ^[L]` be the "error" of layer `L`.

1.  Error at Output Layer (`L`):
    (For Softmax + Cross-Entropy Loss)
    `δ^[L] = a^[L] - y`

2.  Error at Hidden Layer (`l`):
    (Error from the next layer, propagated back)
    `δ^[l] = ( (W^[l+1]).T @ δ^[l+1] )  g'(z^[l])`

3.  Gradient of Cost w.r.t. Biases (`b`):
    (The error of the layer)
    `∂C/∂b^[l] = (1/m)  sum(δ^[l])`

4.  Gradient of Cost w.r.t. Weights (`W`):
    (The error of the layer, scaled by its input)
    `∂C/∂W^[l] = (1/m)  (δ^[l] @ (a^[l-1]).T)`



 6. Gradient Descent (Weight & Bias Update)

For each layer `l` in the network, after calculating the gradients:

-   Update Weights:
    `W^[l] = W^[l] - α  ∂C/∂W^[l]`

-   Update Biases:
    `b^[l] = b^[l] - α  ∂C/∂b^[l]`

(Where `α` is the learning rate)
