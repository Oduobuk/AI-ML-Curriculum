Exercises: CNNs and Image Augmentation

These exercises are designed to build upon the `simple_cnn_tutorial.md`. Use the code from the tutorial as a starting point for these tasks.

 Exercise 1: Modify the CNN Architecture

1.  Add More Layers: Add a second `Conv2D` layer with 64 filters and a `(3, 3)` kernel, followed by another `MaxPooling2D` layer. Place this block after the first `MaxPooling2D` layer in the original model.
2.  Train the New Model: Train the modified model for the same number of epochs.
3.  Analyze the Results:
    How did the number of trainable parameters change? You can see this by running `model.summary()`.
    Did the final test accuracy improve, worsen, or stay the same?
    Did the model take longer to train?

 Exercise 2: Experiment with Hyperparameters

1.  Change the Optimizer: The tutorial uses the `adam` optimizer. Re-compile the original model using a different optimizer, such as `sgd` or `rmsprop`.
2.  Adjust the Learning Rate (Advanced): If you are using `sgd`, you can specify a learning rate. Try `optimizer=tf.keras.optimizers.SGD(learning_rate=0.01)`.
3.  Train and Compare: Train the model with the new optimizer and compare its performance (accuracy and loss) to the original `adam` model.

 Exercise 3: Introduce Image Augmentation

While the MNIST dataset is very clean, image augmentation is a critical skill. Let's practice applying it.

1.  Choose an Augmentation: Using the `image_augmentation_techniques.md` as a guide, choose a geometric transformation (e.g., rotation) from the `albumentations` library.
2.  Create an Augmented Image: Take a single image from the `x_train` dataset and apply the chosen transformation to it.
3.  Visualize the Result: Use a library like `matplotlib` to display the original image and the newly augmented image side-by-side to see the effect of the transformation.

```python
 Example code snippet for visualization
import matplotlib.pyplot as plt
import albumentations as A
import numpy as np

 Select an image from x_train
image_to_augment = x_train[0]

 Define your transformation
transform = A.Rotate(limit=45, p=1)

 Apply the transformation
augmented_image = transform(image=image_to_augment)['image']

 Plot the images
fig, ax = plt.subplots(1, 2)
ax[0].imshow(np.squeeze(image_to_augment), cmap='gray')
ax[0].set_title('Original')
ax[1].imshow(np.squeeze(augmented_image), cmap='gray')
ax[1].set_title('Augmented')
plt.show()
```

 Exercise 4: Adjust Dropout

1.  Change the Dropout Rate: The original model uses a `Dropout` rate of 0.5. Modify this rate to be 0.2.
2.  Train and Observe: Train the model again. Does a lower dropout rate cause the model to overfit more quickly? (Hint: Look at the gap between training accuracy and validation accuracy over epochs).
3.  Remove Dropout: Train the model with the `Dropout` layer completely removed. What happens to the performance and overfitting?