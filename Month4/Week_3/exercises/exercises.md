Exercises: Working with RNNs in Keras

 Exercise 1: Build a Simple RNN

Let's build a `SimpleRNN` to see it in action. We'll use the MNIST dataset again, but treat each image as a sequence of 28 rows, where each row has 28 features (pixels).

1.  Load and Prepare Data: Load the MNIST dataset and normalize the pixel values to be between 0 and 1. No reshaping is needed this time, as the `(28, 28)` shape is perfect for our sequence model.
2.  Build the Model: Create a `Sequential` model with:
    -   A `SimpleRNN` layer with 64 units. The `input_shape` will be `(28, 28)`.
    -   A `Dense` output layer with 10 units and a `softmax` activation.
3.  Compile and Train: Compile the model with `optimizer='adam'`, `loss='sparse_categorical_crossentropy'`, and `metrics=['accuracy']`. Train for 5 epochs.
4.  Evaluate: How does the accuracy of this `SimpleRNN` model compare to the CNN model you built in Week 1?

```python
 Code to get you started
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN
from tensorflow.keras.datasets import mnist

 1. Load and prepare data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

 2. Build the model
model = Sequential([
    SimpleRNN(64, input_shape=(28, 28)),
    Dense(10, activation='softmax')
])

model.summary()

 3. Compile and train (you complete this part)

 4. Evaluate (you complete this part)
```

 Exercise 2: Upgrade to LSTM

Now, let's see the impact of using a more powerful recurrent layer.

1.  Modify the Model: Take the exact model from Exercise 1 and simply replace the `SimpleRNN(64)` layer with an `LSTM(64)` layer.
2.  Train and Evaluate: Train this new LSTM-based model for the same 5 epochs and evaluate its performance on the test set.
3.  Compare: Did the LSTM model achieve a higher accuracy than the `SimpleRNN` model? Was the training time per epoch different?

 Exercise 3: Conceptual Questions

1.  You are building a machine translation system to translate long paragraphs of text from English to French. Why would an LSTM or GRU be a much better choice for this task than a `SimpleRNN`? Refer to the "vanishing gradient problem" in your answer.
2.  You are designing a model for a resource-constrained device (like a low-power microcontroller). You need to process sensor data over time. Your experiments show that an LSTM and a GRU give you very similar accuracy. Which one would you choose to deploy on the device, and why?

 Exercise 4: Stacking Recurrent Layers

Just like with `Dense` and `Conv2D` layers, you can stack recurrent layers. However, there's a special requirement: any recurrent layer that feeds into another recurrent layer must return its full sequence of outputs, not just the final output.

1.  Build a Stacked Model: Create a `Sequential` model with the following architecture:
    -   An `LSTM` layer with 64 units that takes the input. Crucially, set the `return_sequences=True` argument.
    -   A second `LSTM` layer with 64 units. (This layer does not need `return_sequences=True`).
    -   A `Dense` output layer.
2.  Train and Compare: Train this stacked model and compare its performance to the single-layer LSTM from Exercise 2. Does stacking layers help improve performance on this task?