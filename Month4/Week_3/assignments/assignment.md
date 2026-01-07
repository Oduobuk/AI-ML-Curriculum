Assignment: Sentiment Analysis on IMDB Movie Reviews

 Objective

The goal of this assignment is to build a recurrent neural network (RNN) to perform sentiment analysis. You will train a model to classify movie reviews from the IMDB dataset as either "positive" or "negative". This is a classic binary text classification task.

This assignment will require you to use an `Embedding` layer to convert text data into dense vectors, which will then be fed into an `LSTM` or `GRU` layer.

 The IMDB Dataset

The IMDB dataset is a collection of 50,000 movie reviews, split into 25,000 for training and 25,000 for testing. The dataset is pre-processed such that the reviews (sequences of words) are converted into sequences of integers, where each integer represents a specific word in a dictionary.

 Requirements

 1. Data Loading and Preprocessing
- Load the IMDB dataset using `tensorflow.keras.datasets.imdb.load_data()`.
- To keep things manageable, limit the vocabulary to the top 10,000 most frequent words by passing `num_words=10000` to the `load_data` function.
- The reviews have varying lengths. To feed them into a neural network, they must all be the same length. Use the `tensorflow.keras.preprocessing.sequence.pad_sequences` function to pad or truncate all reviews to a fixed length of 256.

 2. Model Architecture
- Build a `Sequential` model in Keras.
- Your model's first layer must be an `Embedding` layer. This layer takes the integer-encoded vocabulary and converts each word into a dense vector of a chosen size (e.g., 128). The `input_dim` should be your vocabulary size (10,000), and the `output_dim` is the size of the embedding.
- After the `Embedding` layer, add either an `LSTM` or a `GRU` layer. You can choose the number of units (e.g., 64).
- Since this is a binary classification problem (positive/negative), the final layer should be a `Dense` layer with a single unit and a `sigmoid` activation function.

 3. Compilation and Training
- Compile your model with the `adam` optimizer, `binary_crossentropy` loss, and `['accuracy']` as the metric.
- Train your model on the padded training data. Use the padded test data as your `validation_data`. Train for at least 5-10 epochs.

 4. Evaluation
- After training, evaluate your model's final performance on the test set.
- Print the final test loss and test accuracy.

 Submission

Submit a Python script (`.py`) or a Jupyter Notebook (`.ipynb`) file containing your complete, executable code that performs all the steps from data loading to final evaluation. Include comments in your code to explain your model architecture and choices.

 Example Code Snippet for Preprocessing

```python
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

 --- 1. Load and Preprocess Data ---
vocab_size = 10000
maxlen = 256

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)

print(len(x_train[0]))  See the length of the first review before padding
print(len(x_train[1]))  See the length of the second review before padding

x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)

print(len(x_train[0]))  See the length after padding
print(len(x_train[1]))  See the length after padding
```

 Grading Criteria

- Correctness (60%): Does the code run without errors? Is the data loaded and preprocessed correctly (including padding)? Is the model built with the required layers (`Embedding`, `LSTM`/`GRU`, `Dense`)?
- Performance (20%): What is the final test accuracy? A good score for this assignment is anything above 85%.
- Code Quality and Comments (20%): Is the code clean, well-organized, and easy to understand?