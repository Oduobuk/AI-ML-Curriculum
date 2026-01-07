Cheatsheet: NLP with Deep Learning

 1. Text Preprocessing Pipeline

The process of cleaning raw text to prepare it for a machine learning model.

| Step | Description | Example |
| :--- | :--- | :--- |
| Lowercasing | Convert all text to lowercase. | "This is Text" -> "this is text" |
| Remove Punctuation | Eliminate characters like `.` `,` `!` `?`. | "Hello, world!" -> "Hello world" |
| Remove Stopwords | Remove common words that carry little meaning (e.g., "a", "the", "is"). | "this is a sentence" -> "sentence" |
| Stemming | Aggressively chop words to their root stem. Fast but can be inaccurate. | "running", "ran" -> "run" |
| Lemmatization | Reduce words to their meaningful base or dictionary form (lemma). Slower but more accurate than stemming. | "better", "best" -> "good" |

---

 2. Text Representation: From Words to Numbers

 Tokenization
The process of breaking text into smaller units (tokens).

- Word Tokenization: Splits text by spaces and punctuation. The most common method.
- Character Tokenization: Splits text into individual characters. Handles misspellings and rare words.
- Subword Tokenization: The modern standard. Breaks rare words into smaller, meaningful units (e.g., "tokenization" -> "token", "ization"). Used by models like BERT and GPT.

 Vectorization (Embeddings)
Converting tokens into numerical vectors.

- Integer Encoding: Assign a unique integer to each unique word in the vocabulary.
- Word Embeddings: Represent each word as a dense vector where similar words have similar vectors.
    - `Embedding` Layer (Keras): A trainable layer that learns embeddings specifically for your dataset during model training.
    - Pre-trained Embeddings (Word2Vec, GloVe): Embeddings learned from a massive corpus of text that can be loaded into your model.

---

 3. Hugging Face Transformers Quick Start

The `transformers` library provides easy access to thousands of powerful, pre-trained NLP models. The simplest way to use them is with the `pipeline` function.

 Using a Pipeline

A pipeline handles all the complex steps (tokenization, model inference, decoding) in a few lines of code.

Sentiment Analysis:
```python
from transformers import pipeline

 This downloads and caches the model on first run
classifier = pipeline("sentiment-analysis")

result = classifier("Hugging Face is a fantastic library!")
 Output: [{'label': 'POSITIVE', 'score': 0.999...}]
```

Fill-Mask (Predicting a missing word):
```python
from transformers import pipeline

unmasker = pipeline("fill-mask")

result = unmasker("The capital of France is [MASK].")
 Output includes predictions like 'paris'
```

Other Common Pipeline Tasks:
- `"ner"` (Named Entity Recognition)
- `"summarization"`
- `"question-answering"`
- `"translation_en_to_fr"`