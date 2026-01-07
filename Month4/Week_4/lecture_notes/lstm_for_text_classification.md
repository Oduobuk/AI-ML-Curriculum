LSTMs for Text Classification and Sentiment Analysis with Keras

Long Short-Term Memory (LSTM) networks are a type of recurrent neural network (RNN) that are particularly well-suited for sequence data like text. They can remember information for long periods, making them effective for tasks like sentiment analysis where the order and context of words are important.

Here's a typical workflow for building a sentiment analysis model using an LSTM with the Keras library:

1.  Data Loading and Preprocessing: Load your dataset, which typically consists of text and corresponding sentiment labels (e.g., "positive," "negative"). Preprocess the text as described in the previous section.

2.  Tokenization and Padding:
    -   Use a `Tokenizer` to convert your text data into sequences of integers. You can limit the vocabulary size to the most frequent words.
    -   Since LSTMs require inputs of a fixed length, you'll need to pad your sequences to ensure they are all the same length.

3.  Model Building: Create a `Sequential` model in Keras and add the following layers:
    -   Embedding Layer: This layer takes the integer-encoded vocabulary and creates an embedding vector for each word.
    -   LSTM Layer: This is the core of the model. You can use a standard `LSTM` layer or a `Bidirectional` LSTM, which can learn from the sequence in both forward and backward directions.
    -   Dense Layer: A fully connected layer is used for the final classification. For binary sentiment analysis, a `sigmoid` activation function is typically used to output a probability between 0 and 1.

4.  Model Compilation and Training:
    -   Compile the model by specifying a loss function (like `binary_crossentropy`), an optimizer (like `adam`), and metrics to monitor (like `accuracy`).
    -   Train the model using the `fit()` method, providing your training data and validation data.

5.  Evaluation and Prediction:
    -   Evaluate the model's performance on your test data.
    -   Use the trained model to predict the sentiment of new, unseen text.

 Keras Code Example

```python
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

 Sample data
texts = ['This is a positive review', 'This is a negative review']
labels = [1, 0]  1 for positive, 0 for negative

 Tokenize the text
tokenizer = Tokenizer(num_words=100)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

 Pad the sequences
padded_sequences = pad_sequences(sequences, maxlen=10)

 Build the model
model = Sequential([
    Embedding(100, 16, input_length=10),
    LSTM(32),
    Dense(1, activation='sigmoid')
])

 Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

 Train the model
model.fit(padded_sequences, labels, epochs=10)

 Predict on new data
new_texts = ['This is a great movie', 'This is a terrible movie']
new_sequences = tokenizer.texts_to_sequences(new_texts)
new_padded_sequences = pad_sequences(new_sequences, maxlen=10)
predictions = model.predict(new_padded_sequences)
print(predictions)
```