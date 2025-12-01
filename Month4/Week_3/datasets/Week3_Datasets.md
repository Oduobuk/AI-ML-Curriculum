# Month 4, Week 3: Datasets for Sequence Modeling (RNNs, LSTMs, GRUs)

This week, as we explore Recurrent Neural Networks (RNNs), LSTMs, and GRUs, our focus shifts to datasets that inherently have a sequential nature. These models are designed to capture dependencies and patterns within ordered data.

Here are some common types of datasets used for sequence modeling tasks:

## 1. IMDB Movie Reviews Dataset

*   **Description:** A classic dataset for **sentiment analysis**. It consists of 50,000 movie reviews from IMDB, labeled as either positive or negative. The reviews are pre-processed, and words are encoded as integers.
*   **Task:** Binary text classification (positive/negative sentiment).
*   **Data Format:** Sequences of integers (word indices).
*   **Use Case:** Ideal for learning to build and train RNNs, LSTMs, or GRUs for text classification. It's a good starting point to understand how these models handle variable-length text inputs.

## 2. Text Corpora for Language Modeling

*   **Description:** Large collections of text (e.g., Wikipedia articles, books, news articles). The goal is often to predict the next word in a sequence or generate new text.
*   **Task:** Language Modeling, Text Generation, Machine Translation.
*   **Data Format:** Raw text, often tokenized into words or sub-word units.
*   **Examples:**
    *   **Penn Treebank (PTB):** A smaller, older corpus often used for academic benchmarks.
    *   **WikiText-2/WikiText-103:** Larger datasets derived from Wikipedia articles, providing more realistic language modeling challenges.
*   **Use Case:** Fundamental for understanding how RNNs learn language structure and generate coherent text.

## 3. Time Series Data

*   **Description:** Any data recorded over successive time intervals. The goal is often to predict future values based on past observations.
*   **Task:** Time Series Forecasting, Anomaly Detection.
*   **Data Format:** Numerical sequences.
*   **Examples:**
    *   **Stock Market Data:** Historical stock prices, volumes, etc.
    *   **Weather Data:** Temperature, humidity, pressure readings over time.
    *   **Sensor Data:** Readings from IoT devices (e.g., temperature, vibration).
    *   **Energy Consumption Data:** Hourly or daily electricity usage.
*   **Use Case:** Demonstrates the ability of RNNs to capture temporal dependencies and make predictions based on historical patterns.

## 4. Speech Datasets

*   **Description:** Collections of audio recordings paired with their corresponding text transcripts.
*   **Task:** Speech Recognition, Speaker Identification.
*   **Data Format:** Audio files (often pre-processed into spectrograms or MFCCs) and text.
*   **Examples:**
    *   **LibriSpeech:** A large corpus of read English speech.
    *   **Common Voice:** Mozilla's open-source, multi-language speech dataset.
*   **Use Case:** Highlights the application of RNNs (especially LSTMs/GRUs) in processing raw audio signals for understanding spoken language.

## 5. Character-Level Text Data

*   **Description:** Instead of words, the model processes text one character at a time.
*   **Task:** Character-level language modeling, text generation, spelling correction.
*   **Data Format:** Sequences of characters.
*   **Use Case:** Useful for understanding the fine-grained sequential processing capabilities of RNNs and for tasks where word-level information might be too coarse (e.g., generating code or highly structured text).

These datasets provide rich opportunities to experiment with different recurrent architectures and observe their strengths and weaknesses in handling various types of sequential information.
