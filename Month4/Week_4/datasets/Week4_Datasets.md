# Month 4, Week 4: Datasets for NLP with Deep Learning

This week, as we delve into Natural Language Processing (NLP) with Deep Learning, we focus on datasets that are rich in textual information. These datasets are crucial for training models that can understand, process, and generate human language.

Here are some common types of datasets used for NLP tasks with deep learning:

## 1. AG News Corpus

*   **Description:** A collection of more than 1 million news articles from over 2000 news sources. It's often used for **text classification** tasks. The dataset is typically pre-processed into a smaller version with 120,000 training samples and 7,600 testing samples, categorized into 4 classes: World, Sports, Business, and Sci/Tech.
*   **Task:** Multi-class text classification.
*   **Data Format:** Text documents and corresponding labels.
*   **Use Case:** Excellent for benchmarking text classification models, including those built with RNNs, LSTMs, GRUs, or even simpler deep learning architectures.

## 2. WikiText Datasets (WikiText-2, WikiText-103)

*   **Description:** These are collections of Wikipedia articles, pre-processed to retain the original casing, punctuation, and numbers. They are significantly larger and more realistic than older language modeling benchmarks.
    *   **WikiText-2:** A smaller version, suitable for quick experimentation.
    *   **WikiText-103:** A much larger version, containing 103 million words, used for training more robust language models.
*   **Task:** Language Modeling (predicting the next word in a sequence), Text Generation.
*   **Data Format:** Raw text.
*   **Use Case:** Ideal for training and evaluating recurrent neural networks (RNNs, LSTMs, GRUs) for language modeling tasks, where the goal is to learn the statistical structure of language.

## 3. WMT (Workshop on Machine Translation) Datasets

*   **Description:** A series of datasets released annually for the WMT shared tasks, primarily focused on **Machine Translation**. These datasets consist of parallel corpora (sentences aligned across two or more languages).
*   **Task:** Machine Translation (e.g., English to German, French to English).
*   **Data Format:** Pairs of sentences in different languages.
*   **Use Case:** Essential for developing and evaluating neural machine translation (NMT) models, which heavily rely on sequence-to-sequence architectures (often built with LSTMs/GRUs and later Transformers).

## 4. Stanford Sentiment Treebank (SST)

*   **Description:** A dataset of movie reviews with fine-grained sentiment labels. Unlike IMDB, it provides sentiment labels not just for entire sentences but also for phrases within sentences, allowing for more nuanced sentiment analysis.
*   **Task:** Fine-grained sentiment analysis, phrase-level sentiment prediction.
*   **Data Format:** Parsed sentences with sentiment labels for sub-phrases.
*   **Use Case:** Useful for training models that require a deeper understanding of sentence structure and compositional semantics for sentiment prediction.

## 5. CoNLL-2003 Dataset

*   **Description:** A widely used dataset for **Named Entity Recognition (NER)**. It consists of news wire text annotated with four types of named entities: persons, locations, organizations, and miscellaneous.
*   **Task:** Named Entity Recognition (sequence tagging).
*   **Data Format:** Sentences with word-level tags indicating named entities.
*   **Use Case:** Excellent for training sequence tagging models (e.g., using Bi-LSTMs) to identify and classify entities in text.

## 6. Common Crawl

*   **Description:** A massive, open repository of web crawl data. It contains petabytes of raw web page data. While not directly a "dataset" in the traditional sense, it's a source from which many large-scale NLP datasets (especially for pre-training word embeddings or large language models) are derived.
*   **Task:** Pre-training word embeddings, large language models.
*   **Data Format:** Raw web page content.
*   **Use Case:** Used by researchers and companies to train foundational NLP models that require truly massive amounts of text data.

These datasets, along with others like SQuAD (for Question Answering) and GLUE/SuperGLUE benchmarks (for evaluating general language understanding), form the backbone of modern NLP research and development.
