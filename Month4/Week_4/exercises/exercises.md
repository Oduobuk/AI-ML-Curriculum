Exercises: Text Preprocessing and Hugging Face Pipelines

 Exercise 1: Manual Text Preprocessing

Practice the fundamental text cleaning steps on the following sentence.

```
raw_sentence = "WOW! The new AI models are SO impressive and powerful."
```

1.  Lowercase: Convert the entire sentence to lowercase.
2.  Remove Punctuation: Remove the `!` and `.` characters.
3.  Tokenize: Split the sentence into a list of individual words (tokens).
4.  Remove Stopwords: Given a simple list of stopwords `['the', 'are', 'and', 'so']`, create a new list of tokens that does not contain these words.

What is the final list of tokens you are left with?

 Exercise 2: Keras Tokenizer and Padding

Use the `tensorflow.keras.preprocessing.text.Tokenizer` to convert text into numerical sequences, a key step for feeding text to models like LSTMs.

```python
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

sentences = [
    'I love deep learning',
    'Keras is a powerful library',
    'Deep learning is fun'
]
```

1.  Initialize and Fit: Create a `Tokenizer` instance and use the `fit_on_texts()` method on the `sentences` list.
2.  Inspect the Word Index: Print the tokenizer's `word_index` attribute. What integer does the word "learning" map to?
3.  Convert to Sequences: Use the `texts_to_sequences()` method to convert the original `sentences` into lists of integers.
4.  Pad the Sequences: The sequences have different lengths. Use the `pad_sequences()` function on your integer sequences to make them all the same length. Print the final padded array.

 Exercise 3: Fun with Hugging Face Pipelines

This exercise is about exploring the power of pre-trained transformers with the easy-to-use `pipeline` API.

1.  Sentiment Analysis:
    -   Create a `sentiment-analysis` pipeline.
    -   Test it on the following two sentences:
        -   `"The plot was predictable and the acting was mediocre."`
        -   `"I was completely captivated by the stunning visuals and compelling story."`
    -   What label and score did the model predict for each sentence?

2.  Fill-Mask:
    -   Create a `fill-mask` pipeline.
    -   Use it to predict the masked word in the sentence: `"The best way to learn programming is by [MASK] code."`
    -   What were the top 3 predictions from the model?

3.  Named Entity Recognition (NER):
    -   Create a `ner` pipeline. NER models are trained to identify and classify named entities like persons, organizations, and locations.
    -   Test it on the sentence: `"Apple Inc. is a technology company headquartered in Cupertino, California, co-founded by Steve Jobs."`
    -   What entities did the model correctly identify? (e.g., `Apple Inc.` as an organization, `Steve Jobs` as a person).