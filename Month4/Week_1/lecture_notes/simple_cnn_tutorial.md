Simple CNN Tutorial for MNIST with Keras

This tutorial provides a step-by-step guide to building a Convolutional Neural Network (CNN) for classifying handwritten digits from the MNIST dataset using Keras.

 1. Import Libraries
First, import the necessary libraries. You'll need `tensorflow` and its `keras` API, along with `numpy`.

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
```

 2. Load and Prepare the Data
The MNIST dataset is included in Keras and can be loaded directly. The dataset is split into 60,000 training images and 10,000 testing images. The images are 28x28 pixel grayscale images of handwritten digits from 0 to 9.

```python
 Load the dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

 Reshape the data to fit the model
 The input shape should be (num_samples, 28, 28, 1)
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)

 Normalize the pixel values from a range of [0, 255] to [0, 1]
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

 Convert class vectors to binary class matrices (one-hot encoding)
num_classes = 10
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)
```

 3. Build the CNN Model
A simple CNN for MNIST can be built using the `Sequential` model in Keras. The architecture will consist of convolutional layers, pooling layers, and dense layers.

    Conv2D: These layers apply convolution filters to the input, detecting patterns like edges.
    MaxPooling2D: This downsamples the feature maps, reducing the spatial dimensions and the number of parameters.
    Flatten: This layer converts the 2D feature maps into a 1D vector to be fed into the dense layers.
    Dense: A fully connected layer.
    Dropout: A regularization technique to prevent overfitting.
    Softmax: The final activation function for multi-class classification.

```python
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])
```

 4. Compile the Model
Before training, you need to configure the model for training by compiling it. You'll specify the optimizer, loss function, and metrics to monitor.

```python
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
```

 5. Train the Model
Now, you can train the model using the `fit` method on the training data. You'll specify the batch size and the number of epochs (iterations over the entire dataset).

```python
batch_size = 128
epochs = 12

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
```

 6. Evaluate the Model
Finally, evaluate the performance of the trained model on the test set.

```python
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
```
This simple CNN architecture can achieve a high accuracy (often around 99%) on the MNIST dataset.