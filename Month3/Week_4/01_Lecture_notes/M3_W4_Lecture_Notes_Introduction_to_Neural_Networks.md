 Month 3, Week 4: Introduction to Neural Networks

 1. Conceptual Foundation: What is a Neural Network?

Inspired by the human brain, a neural network is a computational model that learns to perform tasks by considering examples, generally without being programmed with task-specific rules. We'll explore this by considering a classic problem: handwritten digit recognition.

 1.1. The Structure: Layers of Neurons

Imagine a program that takes a 28x28 pixel image and tells you which digit (0-9) it represents. How would you even start? This is where the layered structure of a neural network shines.

- Neuron: For now, think of a neuron as simply a container that holds a number between 0 and 1. This number is called the neuron's activation. A high activation (close to 1) means the neuron is "lit up," while a low activation (close to 0) means it's inactive.

- Input Layer: This is the first layer of the network. For our 28x28 pixel image, we would have 784 neurons in the input layer (one for each pixel). The activation of each neuron corresponds to the grayscale value of its pixel, where 0 is black and 1 is white.

- Output Layer: This is the final layer. For digit recognition, it would have 10 neurons, one for each digit (0 through 9). After the network processes an image, the activations in this layer represent the network's confidence. The neuron with the highest activation is the network's "answer."

- Hidden Layers: These are the layers that exist between the input and output. They are the "black box" where the real magic happens. The network learns to use these layers to recognize increasingly complex patterns. In our example, we might use two hidden layers, each with 16 neurons.

![Simple Network Structure](https://i.imgur.com/BDSm29V.png)

 1.2. The "Why": Layers of Abstraction

Why this layered structure? Because complex patterns are often built from simpler ones.

1.  From Pixels to Edges: The first hidden layer might learn to recognize small, simple patterns from the pixels, like tiny edges or curves. For example, one neuron might become highly active only when it sees a short horizontal edge in a specific part of the image.

2.  From Edges to Patterns: The second hidden layer might take the outputs from the "edge detector" layer and learn to recognize more complex patterns by combining them. For instance, a neuron in this layer could learn to recognize a circle by seeing if the neurons for a top-curve, bottom-curve, left-curve, and right-curve are all active.

3.  From Patterns to Digits: The final output layer takes the outputs from the "pattern detector" layer and learns which combinations of patterns correspond to which digits. A "9" is a top-loop pattern combined with a vertical-line pattern. An "8" is a top-loop combined with a bottom-loop.

This hierarchical breakdown of a complex problem into simpler, layered sub-problems is the core intuition behind the power of deep learning.

 1.3. The "How": Connecting the Layers

How does the activation of one layer determine the activation of the next? 

Let's focus on a single neuron in the first hidden layer. It needs to decide its activation based on the 784 pixel values from the input layer.

- Weights: We assign a "weight" to each connection from an input neuron to our hidden neuron. A weight is just a number. A positive weight (green) means the input pixel contributes to activating the hidden neuron. A negative weight (red) means the input pixel contributes to deactivating it.

- Weighted Sum: The first step is to calculate a weighted sum of all the input activations. You take each input neuron's activation, multiply it by its corresponding weight, and sum them all up. If we want our neuron to detect an edge in a specific region, we could set the weights for pixels in that region to be positive and the weights for surrounding pixels to be negative.

- Bias: After the weighted sum, we add a single number called a bias. The bias acts as a threshold. For example, if a bias is -10, it means the weighted sum must be greater than 10 for the neuron to start becoming meaningfully active.

- Activation Function (Squishification): The result of the weighted sum + bias can be any number. To get it into our desired 0-to-1 range, we pass it through an activation function. A classic choice is the Sigmoid function, which squishes any real number into the (0, 1) interval.

So, the full process for a single neuron is:
`activation = sigmoid((w1a1 + w2a2 + ...) + bias)`

Every neuron in the hidden layer does this, each with its own unique set of weights and its own bias. This allows each neuron to specialize in detecting a different pattern.

 1.4. Compact Representation: The Matrix Math

Writing out those sums for every neuron is cumbersome. Linear algebra gives us a much cleaner way to represent this.

1.  Organize the activations of a layer into a column vector `a`.
2.  Organize all the weights connecting to the next layer into a matrix `W`. Each row of `W` contains the weights for one neuron in the next layer.
3.  Organize the biases for the next layer into a column vector `b`.

The activations of the next layer can then be calculated with a single, neat equation:

`next_activations = sigmoid(W  a + b)`

This is not only easier to write but also much faster for computers to execute, as matrix multiplication is highly optimized.

---

 2. The Engine of Learning: Backpropagation

How does the network "learn" the correct values for its 13,000+ weights and biases? It does so by a process called backpropagation, which is essentially a method for intelligently adjusting the parameters to reduce error.

 2.1. The Cost Function

First, we need a way to tell the network when it's wrong. We define a cost function (or loss function). For a single training example (e.g., an image of a "3"), we do the following:
1.  Feed the image through the network and get the 10 output activations.
2.  Compare the output activations to the desired output (where the "3" neuron is 1 and all others are 0).
3.  The cost is the sum of the squared differences between the network's output and the desired output. `Cost = (a₀ - y₀)² + (a₁ - y₁)² + ...`
4.  The total cost for the network is the average of these individual costs over all training examples in our dataset.

The goal of training is to minimize this cost function. This is an optimization problem.

 2.2. Gradient Descent: Finding the "Downhill" Direction

Imagine the cost function as a complex, high-dimensional landscape. Our current set of weights and biases places us somewhere on this landscape. We want to find the deepest valley.

Gradient Descent is the algorithm for this. The gradient of the cost function is a vector that points in the direction of the steepest "uphill" ascent. Therefore, if we take a small step in the opposite direction of the gradient, we will move downhill, reducing the cost.

The core of backpropagation is to calculate this gradient—specifically, the partial derivative of the cost function with respect to every single weight and bias in the network (`∂C/∂w`, `∂C/∂b`).

 2.3. The Chain Rule: Propagating Error Backwards

How does a single weight `w` deep inside the network affect the final cost `C`? It does so through a chain of influences:
1.  A change in a weight `w` causes a change in the weighted sum `z` of the next neuron.
2.  A change in `z` causes a change in that neuron's activation `a`.
3.  A change in `a` propagates through the subsequent layers, ultimately causing a change in the final cost `C`.

The Chain Rule from calculus lets us quantify this. The sensitivity of the cost to the weight (`∂C/∂w`) is the product of the sensitivities at each step in this chain:

`∂C/∂w = (∂z/∂w)  (∂a/∂z)  (∂C/∂a)`

- `∂z/∂w`: How much does the weighted sum change with the weight? This turns out to be simply the activation of the neuron from the previous layer. This gives rise to the famous Hebbian postulate: "neurons that fire together, wire together." The weight adjustment is most significant when the input neuron is highly active.
- `∂a/∂z`: How much does the activation change with the weighted sum? This is simply the derivative of the activation function (e.g., the sigmoid).
- `∂C/∂a`: How much does the cost change with the neuron's activation? This is the most complex part. This term itself depends on the weights and activations of the next layers.

This is why the algorithm is called backpropagation. To calculate the gradient for the weights of a layer, you first need to calculate the error influence (`∂C/∂a`) from the layer in front of it. You start at the output layer and work your way backward, propagating the error gradient through the network.

---

 3. Practical Implementation: A Neural Network from Scratch in Python

Let's translate these concepts into a working Python script using only the `numpy` library. We will build a network to classify the MNIST handwritten digits.

 3.1. The Code Structure

We'll create a `BackPropagationNetwork` class to encapsulate the logic.

```python
import numpy as np

class BackPropagationNetwork:
    """A simple back-propagation neural network."""

    def __init__(self, layer_size):
        """
        Initializes the network.
        :param layer_size: A tuple of integers representing the number of neurons in each layer.
                           e.g., (784, 10, 10) for an input layer of 784, a hidden layer of 10,
                           and an output layer of 10.
        """
         Layer Information
        self.layer_count = len(layer_size) - 1
        self.shape = layer_size

         Internal data from the last run
        self._layer_input = []
        self._layer_output = []
        
         Weight and Bias arrays
        self.weights = []

         Initialize the weight arrays
         We iterate through pairs of layers (e.g., (784, 10) and (10, 10))
        for (l1, l2) in zip(layer_size[:-1], layer_size[1:]):
             Create a matrix of random weights.
             The dimensions are (neurons_in_next_layer, neurons_in_previous_layer + 1)
             The +1 is to account for the bias node.
            self.weights.append(
                np.random.normal(scale=0.1, size=(l2, l1 + 1))
            )

    def run(self, input_data):
        """
        Runs the network on a given input.
        :param input_data: A numpy array of shape (n, m) where n is the number of inputs
                           and m is the number of samples.
        :return: The network's output.
        """
        num_samples = input_data.shape[1]

         Clear previous run's data
        self._layer_input = []
        self._layer_output = []

         Run it!
        for i in range(self.layer_count):
             Determine the input for the current layer
            if i == 0:
                 For the first layer, the input is the raw data plus a bias row
                layer_input = np.vstack([input_data, np.ones([1, num_samples])])
            else:
                 For subsequent layers, the input is the output of the previous layer plus a bias row
                layer_input = np.vstack([self._layer_output[-1], np.ones([1, num_samples])])

            self._layer_input.append(layer_input)
            
             Calculate the weighted sum (dot product)
            weighted_sum = self.weights[i] @ layer_input
            
             Apply the sigmoid activation function
            layer_output = self.sigmoid(weighted_sum)
            self._layer_output.append(layer_output)

        return self._layer_output[-1]

     The sigmoid activation function
    def sigmoid(self, x, derivative=False):
        if derivative:
            return x  (1 - x)
        return 1 / (1 + np.exp(-x))

 Main execution block to test the class
if __name__ == "__main__":
     Create a test network for a 2-input, 2-hidden, 1-output configuration
    bpn = BackPropagationNetwork((2, 2, 1)) 
    
    print("Network Shape:", bpn.shape)
    print("\nInitial Weights:")
    for i, w in enumerate(bpn.weights):
        print(f"Layer {i+1} -> {i+2}:")
        print(w)
        print("-"  20)

     Create a dummy input
    dummy_input = np.array([[1.5], [0.5]])
    
     Run the network
    output = bpn.run(dummy_input)
    print("\nOutput for dummy input:", output)

```

 3.2. Explanation of the Code

1.  `__init__(self, layer_size)`:
       The constructor takes a tuple `layer_size` that defines the network's architecture.
       It initializes the `weights` list. For each connection between layers (e.g., layer 1 to 2), it creates a NumPy matrix of small, random numbers.
       The matrix dimensions are `(next_layer_neurons, prev_layer_neurons + 1)`. The `+1` is a clever trick to handle the bias. By adding a "dummy" input neuron that is always 1, the weight connected to it acts exactly like a bias term, simplifying the math.

2.  `run(self, input_data)`:
       This is the forward propagation step.
       It takes an `input_data` array and passes it through the network layer by layer.
       In each layer, it first adds the bias term (by stacking a row of ones).
       Then, it performs the matrix multiplication (`@` operator) between the layer's weight matrix and its input vector.
       Finally, it applies the `sigmoid` activation function to the result.
       It stores the inputs and outputs of each layer, which will be crucial for backpropagation.
       It returns the final layer's output.

3.  `sigmoid(self, x, derivative=False)`:
       A helper function that implements the sigmoid activation.
       It also includes the logic to calculate the derivative of the sigmoid, which will be needed for training.

This script sets up the structure and the forward-pass mechanism. The next logical step, which involves creating a `train` method to implement backpropagation and update the weights, is the core of the learning process. It would involve calculating the error at the output and using the chain rule and the stored `_layer_input` and `_layer_output` values to propagate this error backward, adjusting weights along the way.
