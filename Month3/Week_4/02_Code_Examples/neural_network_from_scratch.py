import numpy as np
import pickle
import gzip

  Activation Functions 

def sigmoid(x, derivative=False):
    """
    Sigmoid activation function.
    If derivative is True, it returns the derivative of the sigmoid function.
    """
    if derivative:
        return x  (1 - x)
    return 1 / (1 + np.exp(-x))

def relu(x, derivative=False):
    """
    Rectified Linear Unit (ReLU) activation function.
    """
    if derivative:
        return np.where(x > 0, 1, 0)
    return np.maximum(0, x)

def softmax(x):
    """
    Softmax activation function for the output layer.
    It converts logits into probabilities.
    """
     Subtract max for numerical stability
    exps = np.exp(x - np.max(x, axis=0, keepdims=True))
    return exps / np.sum(exps, axis=0, keepdims=True)

  The Neural Network Class 

class BackPropagationNetwork:
    """A back-propagation neural network."""

    def __init__(self, layer_sizes, activation_funcs=None):
        """
        Initializes the network.
        :param layer_sizes: A tuple of integers for neuron counts in each layer.
        :param activation_funcs: A list of activation functions for hidden layers. Defaults to sigmoid.
        """
        self.layer_count = len(layer_sizes) - 1
        self.shape = layer_sizes
        
        if activation_funcs and len(activation_funcs) == len(layer_sizes) - 2:
            self.activation_funcs = [globals()[func] for func in activation_funcs]
        else:
            self.activation_funcs = [sigmoid]  (len(layer_sizes) - 2)

         Internal data from the last run
        self._layer_input = []
        self._layer_output = []
        
         Weight and Bias arrays
        self.weights = []
        for (l1, l2) in zip(layer_sizes[:-1], layer_sizes[1:]):
             He initialization for ReLU, Xavier/Glorot for sigmoid/tanh
             A small random normal initialization is a safe general bet.
            self.weights.append(np.random.normal(scale=np.sqrt(2. / l1), size=(l2, l1 + 1)))

    def run(self, input_data):
        """
        Performs a forward pass through the network.
        :param input_data: A numpy array of shape (n_features, n_samples).
        :return: The network's output.
        """
        num_samples = input_data.shape[1]
        self._layer_input = []
        self._layer_output = []

        for i in range(self.layer_count):
            if i == 0:
                layer_input = np.vstack([input_data, np.ones([1, num_samples])])
            else:
                layer_input = np.vstack([self._layer_output[-1], np.ones([1, num_samples])])

            self._layer_input.append(layer_input)
            weighted_sum = self.weights[i] @ layer_input
            
             Determine activation function
            if i < self.layer_count - 1:  Hidden layers
                activation_func = self.activation_funcs[i] if i < len(self.activation_funcs) else sigmoid
                layer_output = activation_func(weighted_sum)
            else:  Output layer
                layer_output = softmax(weighted_sum)

            self._layer_output.append(layer_output)

        return self._layer_output[-1]

    def train(self, input_data, labels, learning_rate, epochs):
        """
        Trains the network using backpropagation.
        :param input_data: Training data.
        :param labels: One-hot encoded training labels.
        :param learning_rate: The learning rate (alpha).
        :param epochs: The number of training iterations.
        """
        num_samples = input_data.shape[1]

        for epoch in range(epochs):
             Forward pass
            output = self.run(input_data)

              Backward pass (Backpropagation) 
            deltas = []

             1. Calculate error at the output layer
             For softmax with cross-entropy loss, the delta is simply (predictions - true_labels)
            error = output - labels
            deltas.append(error)

             2. Propagate error to hidden layers
            for i in range(self.layer_count - 2, -1, -1):
                 Get activation function for the current layer
                activation_func = self.activation_funcs[i] if i < len(self.activation_funcs) else sigmoid
                
                 Error from next layer  derivative of current layer's activation
                 Note: self._layer_output[i] is the output of layer i+1
                 The weighted sum for this layer is needed for the derivative
                weighted_sum_for_deriv = self.weights[i+1].T @ deltas[-1]
                
                 We need the derivative of the activation of layer i
                 The output of layer i is self._layer_output[i]
                derivative = activation_func(self._layer_output[i], derivative=True)
                
                 Remove bias influence for backpropagation
                delta = (self.weights[i+1].T[:-1, :] @ deltas[-1])  derivative
                deltas.append(delta)

            deltas.reverse()

             3. Update weights
            for i in range(len(self.weights)):
                layer_input = self._layer_input[i]
                delta = deltas[i]
                
                 Gradient is delta dotted with the input that created it
                gradient = delta @ layer_input.T / num_samples
                
                 Update weight
                self.weights[i] -= learning_rate  gradient
            
             Print progress
            if (epoch + 1) % 100 == 0:
                loss = -np.sum(labels  np.log(output + 1e-9)) / num_samples  Categorical Cross-Entropy
                accuracy = np.mean(np.argmax(output, axis=0) == np.argmax(labels, axis=0))
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")


  Data Loading and Preparation 
def load_mnist():
    """Loads the MNIST dataset."""
    with gzip.open('/home/david/Downloads/mnist.pkl.gz', 'rb') as f:
        train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
    return train_set, valid_set, test_set

def one_hot_encode(y, num_classes=10):
    """One-hot encodes a vector of labels."""
    return np.eye(num_classes)[y].T

if __name__ == '__main__':
     Load data
    try:
        train_set, valid_set, test_set = load_mnist()
        
         Prepare data
        train_images, train_labels_raw = train_set
        train_images = train_images.T  Shape: (784, 50000)
        train_labels = one_hot_encode(train_labels_raw)  Shape: (10, 50000)

          Network Setup and Training 
        print("Initializing and training the network...")
         A 784-input, 100-hidden, 10-output network
        network = BackPropagationNetwork((784, 100, 10), activation_funcs=['relu'])
        
         Train the network
        network.train(train_images, train_labels, learning_rate=0.1, epochs=1000)

          Evaluation 
        print("\nEvaluating on test data...")
        test_images, test_labels_raw = test_set
        test_images = test_images.T
        test_labels = one_hot_encode(test_labels_raw)

        predictions = network.run(test_images)
        test_accuracy = np.mean(np.argmax(predictions, axis=0) == test_labels_raw)
        print(f"Test Accuracy: {test_accuracy:.4f}")

    except FileNotFoundError:
        print("\n")
        print("MNIST dataset not found at '/home/david/Downloads/mnist.pkl.gz'")
        print("Please download it from http://deeplearning.net/data/mnist/mnist.pkl.gz")
        print("")
