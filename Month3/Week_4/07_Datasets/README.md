 Dataset for Week 4: The MNIST Database

The primary dataset for this week's topic on neural networks is the MNIST Database of Handwritten Digits.

This dataset is famous within the machine learning community and is often referred to as the "hello world" of deep learning. It's an excellent dataset for training and testing your first neural network.

 Dataset Details

-   Content: A large collection of 28x28 pixel grayscale images of handwritten digits (0 through 9).
-   Size:
    -   Training Set: 60,000 images.
    -   Test Set: 10,000 images.
-   Format: The images are normalized and centered.

 How to Get the Dataset

You do not need to manually download image files. The dataset is available through many machine learning libraries and as a pre-packaged file.

 Option 1: Using the Pre-packaged Pickle File (Recommended for the Code Example)

The code example in `02_Code_Examples` is configured to use a specific version of MNIST that has been pre-packaged into a Python `pickle` file. This is the most straightforward way to get the code running.

1.  Download the file:
    -   URL: [http://deeplearning.net/data/mnist/mnist.pkl.gz](http.://deeplearning.net/data/mnist/mnist.pkl.gz)

2.  Save the file:
    -   Place the downloaded `mnist.pkl.gz` file into your `/home/david/Downloads/` directory. The script is hardcoded to look for it there.

3.  Run the script:
    -   The `neural_network_from_scratch.py` script will automatically load and decompress this file.

 Option 2: Using a Deep Learning Framework (e.g., TensorFlow/Keras)

If you are working in an environment with TensorFlow or Keras, you can load the dataset with a single command.

```python
import tensorflow as tf

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

 The data will be in the form of numpy arrays.
 You may need to reshape and normalize it to match the format used in our from-scratch network.
 For example:
 x_train = x_train.reshape(60000, 784).T / 255.0
```

 Option 3: Using PyTorch

PyTorch's `torchvision` library also provides a convenient way to download and load MNIST.

```python
import torchvision
import torchvision.transforms as transforms

 This will download the dataset to the './data' directory
train_dataset = torchvision.datasets.MNIST(root='./data',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='./data',
                                          train=False,
                                          transform=transforms.ToTensor())
```

For this week's assignment and code examples, Option 1 is the most direct path. Options 2 and 3 are good to know for future projects.
