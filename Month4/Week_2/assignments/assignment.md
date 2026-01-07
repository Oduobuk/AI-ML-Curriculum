Assignment: Transfer Learning for Cat vs. Dog Classification

 Objective

In this assignment, you will use the powerful technique of transfer learning to build a high-performance image classifier. Your task is to train a model that can accurately distinguish between images of cats and dogs.

You will use a pre-trained model and apply both feature extraction and fine-tuning to see how they perform on a new dataset.

 The Dataset

You will use the "Cats vs. Dogs" dataset, which is available through TensorFlow Datasets. This is a classic binary image classification problem.

 Requirements

 Part 1: Feature Extraction

1.  Load the Data: Load the `cats_vs_dogs` dataset from `tensorflow_datasets`. The dataset should be split into training and testing sets.
2.  Preprocess the Data:
    -   Resize all images to a consistent size suitable for a pre-trained model (e.g., `(160, 160)` for MobileNetV2).
    -   Normalize the pixel values to be in the `[0, 1]` range.
3.  Load a Pre-trained Base Model:
    -   Load a pre-trained model from `tf.keras.applications`, such as `MobileNetV2` or `VGG16`.
    -   Instantiate the model with `weights='imagenet'` and `include_top=False`.
    -   Freeze the entire convolutional base by setting `layer.trainable = False`.
4.  Create Your New Model:
    -   Create a `Sequential` model that starts with the frozen pre-trained base.
    -   Add a `GlobalAveragePooling2D` layer after the base to flatten the features.
    -   Add a `Dense` output layer with a single unit and a `sigmoid` activation function for binary classification.
5.  Compile and Train:
    -   Compile the model with the `adam` optimizer, `binary_crossentropy` loss, and `['accuracy']` metric.
    -   Train the model on your training data.
6.  Evaluate: Report the final accuracy on the test set.

 Part 2: Fine-Tuning

1.  Unfreeze Layers: Take your trained model from Part 1. Unfreeze the top layers of the pre-trained base. A good starting point is to unfreeze the last 20-30% of the layers.
    -   Example: `for layer in base_model.layers[-30:]: layer.trainable = True`
2.  Re-compile for Fine-Tuning:
    -   Re-compile the model. It is crucial to use a very low learning rate to prevent overfitting and destroying the learned weights. Use an optimizer like `tf.keras.optimizers.Adam(learning_rate=0.0001)`.
3.  Continue Training (Fine-tune):
    -   Continue training the model for a few more epochs.
4.  Evaluate Again: Report the new final accuracy on the test set after fine-tuning.

 Submission

Submit a Python script (`.py`) or a Jupyter Notebook (`.ipynb`) file containing your complete, executable code for both Part 1 and Part 2.

Include a summary section (using markdown comments) at the end of your notebook that answers the following questions:
1.  What was the test accuracy of your model after the initial training (Part 1: Feature Extraction)?
2.  What was the test accuracy of your model after fine-tuning (Part 2)?
3.  Did fine-tuning improve your model's performance? Briefly explain why you think this was the case.

 Grading Criteria

- Part 1 Implementation (40%): Correctly implements the feature extraction workflow.
- Part 2 Implementation (40%): Correctly implements the fine-tuning workflow with a low learning rate.
- Results and Analysis (20%): Reports the accuracies and provides a clear, concise analysis comparing the two methods.