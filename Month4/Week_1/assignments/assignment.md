Assignment: CIFAR-10 Image Classification

 Objective

The goal of this assignment is to build, train, and evaluate a Convolutional Neural Network (CNN) to classify images from the CIFAR-10 dataset. This will require you to apply the concepts learned in Week 1, from data preparation to model architecture design.

 The CIFAR-10 Dataset

CIFAR-10 is a widely-used benchmark dataset in computer vision. It consists of 60,000 32x32 color images in 10 different classes, with 6,000 images per class. The classes are:

1.  airplane
2.  automobile
3.  bird
4.  cat
5.  deer
6.  dog
7.  frog
8.  horse
9.  ship
10. truck

 Requirements

 1. Data Loading and Preprocessing
- Load the CIFAR-10 dataset using `tensorflow.keras.datasets.cifar10.load_data()`.
- Normalize the pixel values of the training and testing images to be between 0 and 1.
- One-hot encode the training and testing labels.
- Important: The images in CIFAR-10 are color images, so their shape will be `(32, 32, 3)`. Make sure your model's input layer is configured to handle this shape.

 2. Model Architecture
- Design a CNN architecture using a `Sequential` model in Keras.
- Your model must contain at least:
    - Two `Conv2D` layers.
    - Two `MaxPooling2D` layers.
    - One `Flatten` layer.
    - Two `Dense` layers.
    - At least one `Dropout` layer for regularization.
- You are free to choose the number of filters, kernel sizes, and number of units in the dense layers. Experiment to find a good architecture.

 3. Compilation and Training
- Compile your model using an appropriate optimizer (e.g., `adam`) and loss function (`categorical_crossentropy`).
- Train your model on the CIFAR-10 training data. Train for at least 10 epochs.
- Use the test data as your `validation_data` to monitor performance during training.

 4. Evaluation
- After training is complete, evaluate your model's final performance on the test set.
- Print the final test loss and test accuracy.

 Submission

Submit a Python script (`.py`) or a Jupyter Notebook (`.ipynb`) file containing your complete, executable code. The code should perform all the steps from data loading to final evaluation. Include comments in your code to explain your model architecture and design choices.

 Grading Criteria

- Correctness (60%): Does the code run without errors? Is the data preprocessed correctly? Is the model built and trained as required?
- Performance (20%): What is the final test accuracy of your model? (Note: State-of-the-art is >95%, but a good score for this assignment is anything above 65-70%).
- Code Quality and Comments (20%): Is the code clean, well-organized, and easy to understand? Are there comments explaining your choices?

Good luck!