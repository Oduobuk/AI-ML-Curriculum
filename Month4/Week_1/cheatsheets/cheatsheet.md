Cheatsheet: CNNs for Image Classification

 Core CNN Layers

| Layer Type | Purpose | Keras Layer |
| :--- | :--- | :--- |
| Convolutional | Extracts features (edges, textures) by applying filters. | `Conv2D` |
| Activation | Introduces non-linearity to learn complex patterns. | `activation='relu'` |
| Pooling | Downsamples feature maps to reduce dimensions and prevent overfitting. | `MaxPooling2D` |
| Flatten | Converts 2D feature maps into a 1D vector for dense layers. | `Flatten` |
| Dense | A standard fully-connected neural network layer. | `Dense` |
| Dropout | Regularization technique to prevent overfitting by randomly dropping units. | `Dropout` |
| Output | Produces the final classification probability distribution. | `Dense(activation='softmax')` |

 Keras CNN Workflow

A typical workflow for building a CNN in Keras involves five main steps:

 1. Load and Prepare Data
- Load the dataset (e.g., from `keras.datasets`).
- Reshape data to include the channel dimension (e.g., `(samples, height, width, channels)`).
- Normalize pixel values to a range of [0, 1].
- One-hot encode the target labels using `keras.utils.to_categorical`.

 2. Build the Model
- Initialize a `Sequential` model.
- Add a stack of `Conv2D` and `MaxPooling2D` layers.
- `Flatten` the output from the convolutional base.
- Add `Dense` layers for classification.
- Use `Dropout` for regularization.
- Add a final `Dense` layer with a `softmax` activation for the output.

 3. Compile the Model
- Use the `model.compile()` method.
- Optimizer: `adam` is a common and effective choice.
- Loss Function: `categorical_crossentropy` is used for multi-class classification.
- Metrics: `['accuracy']` is used to monitor the classification accuracy during training.

 4. Train the Model
- Use the `model.fit()` method.
- Provide training data (`x_train`, `y_train`).
- Set the `batch_size` (number of samples per gradient update).
- Set the `epochs` (number of times to iterate over the entire training dataset).
- Provide `validation_data` to monitor performance on unseen data.

 5. Evaluate the Model
- Use the `model.evaluate()` method on the test set (`x_test`, `y_test`).
- This returns the loss and accuracy on the test data, indicating the model's performance.