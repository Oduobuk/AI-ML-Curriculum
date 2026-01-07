Assignment: Fine-Tuning a Transformer for Sentiment Analysis

 Objective

In last week's assignment, you built an LSTM to classify IMDB movie reviews. In this assignment, you will tackle the same problem but with a much more powerful and modern approach: fine-tuning a pre-trained Transformer model using the Hugging Face ecosystem.

This will demonstrate the remarkable performance and efficiency of using large, pre-trained language models.

 The Hugging Face Ecosystem

You will use three key components from the Hugging Face library:
1.  `datasets`: To load the IMDB dataset.
2.  `transformers.AutoTokenizer`: To load the correct tokenizer for our pre-trained model.
3.  `transformers.AutoModelForSequenceClassification`, `TrainingArguments`, and `Trainer`: To load the model and handle the entire fine-tuning and evaluation process.

 Requirements

 1. Load the Dataset
- Use the `datasets.load_dataset` function to load the `"imdb"` dataset.
- The library will handle splitting it into `train` and `test` sets for you.

 2. Tokenization
- Choose a pre-trained model to fine-tune. A great choice for this task is `"distilbert-base-uncased"` because it's effective yet smaller and faster to train than models like BERT or RoBERTa.
- Load the corresponding tokenizer using `AutoTokenizer.from_pretrained("distilbert-base-uncased")`.
- Create a function that takes a batch of examples and uses the tokenizer to convert the `"text"` field into `"input_ids"`, `"attention_mask"`, etc.
- Use the `.map()` method of the dataset object to apply this tokenization function to the entire dataset.

 3. Fine-Tuning the Model
- Load the pre-trained model using `AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)`. We specify `num_labels=2` because we have two classes: positive and negative.
- Define your training hyperparameters using the `TrainingArguments` class. You only need to train for a small number of epochs (e.g., 1 or 2) to get great results. Set a reasonable batch size and specify an output directory.
- Create a `Trainer` instance, passing it your model, training arguments, and the tokenized training and test datasets.
- Call the `trainer.train()` method to start the fine-tuning process.

 4. Evaluation
- After training is complete, call the `trainer.evaluate()` method.
- The `Trainer` will automatically run evaluation on your test set and print the results, including the final accuracy.

 Submission

Submit a Python script (`.py`) or a Jupyter Notebook (`.ipynb`) file containing your complete, executable code.

Include a summary section (using markdown comments) at the end of your notebook that answers the following questions:
1.  What was the final test accuracy of your fine-tuned Transformer model?
2.  How does this accuracy compare to the accuracy you achieved with your LSTM model from Week 3's assignment?
3.  How did the training time compare? Was fine-tuning the Transformer faster or slower per epoch than training the LSTM from scratch?

 Example Code Structure

```python
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

 1. Load Dataset
imdb_dataset = load_dataset("imdb")

 2. Tokenization
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = imdb_dataset.map(tokenize_function, batched=True)

 3. Load Model
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

 4. Set up Training
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,  Just one epoch is often enough for great results
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    logging_dir='./logs',
    evaluation_strategy="epoch"  Evaluate at the end of each epoch
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
)

 5. Train
trainer.train()

 6. Evaluate
evaluation_results = trainer.evaluate()
print(evaluation_results)
```

 Grading Criteria
- Correctness (70%): Does the code correctly load the data, tokenizer, and model? Does it successfully use the `Trainer` API to fine-tune and evaluate?
- Analysis (30%): Do you provide a clear comparison of the results (accuracy, training time) between the Transformer and the previous LSTM model?