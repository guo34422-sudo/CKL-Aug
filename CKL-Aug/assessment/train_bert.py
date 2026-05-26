from datetime import datetime
import time
import argparse

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)


parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--test_path", type=str, required=True)
parser.add_argument("--model_path", type=str, required=True)
parser.add_argument("--result_txt_path", type=str, required=True)
parser.add_argument("--num_train_epochs", type=int, default=5)
parser.add_argument("--batch_size", type=int, default=8)

args = parser.parse_args()

data_path = args.data_path
test_path = args.test_path
model_path = args.model_path
result_txt_path = args.result_txt_path
num_train_epochs = args.num_train_epochs
batch_size = args.batch_size

start_time = time.time()
start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

df = pd.read_excel(data_path)
df = df.dropna()
df = df[["sentence", "label"]]

dataset = Dataset.from_pandas(df)

test_df = pd.read_excel(test_path)
test_df = test_df.dropna()
test_df = test_df[["sentence", "label"]]

test_dataset = Dataset.from_pandas(test_df)

tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path, num_labels=2)


def preprocess_function(examples):
    tokenized_inputs = tokenizer(
        examples["sentence"],
        truncation=True,
        padding="max_length",
        max_length=512
    )
    tokenized_inputs["labels"] = examples["label"]
    return tokenized_inputs


dataset = dataset.map(preprocess_function, batched=True)

train_data = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = train_data["train"]
val_dataset = train_data["test"]


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    accuracy = accuracy_score(labels, preds)

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="macro",
        zero_division=0
    )

    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="weighted",
        zero_division=0
    )

    return {
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted
    }


training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    warmup_ratio=0.1,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=50,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_accuracy",
    greater_is_better=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
)

trainer.train()

test_dataset = test_dataset.map(preprocess_function, batched=True)

predictions = trainer.predict(test_dataset)
true_labels = test_df["label"].values
metrics = compute_metrics((predictions.predictions, true_labels))

print(data_path)

for metric_name, metric_value in metrics.items():
    print(f"{metric_name}: {metric_value:.4f}")

end_time = time.time()
end_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
elapsed_time = end_time - start_time

with open(result_txt_path, "a", encoding="utf-8") as f:
    f.write(f"start_datetime: {start_datetime}\n")
    f.write(f"end_datetime: {end_datetime}\n")
    f.write(f"elapsed_time: {elapsed_time:.2f} seconds\n")

    f.write("\n========== data_info ==========\n")
    f.write(f"data_path: {data_path}\n")
    f.write(f"test_path: {test_path}\n")
    f.write(f"model_path: {model_path}\n")
    f.write(f"len(df): {len(df)}\n")
    f.write(f"len(train_dataset): {len(train_dataset)}\n")
    f.write(f"len(val_dataset): {len(val_dataset)}\n")
    f.write(f"len(test_dataset): {len(test_dataset)}\n")

    f.write("\n========== test result ==========\n")
    for metric_name, metric_value in metrics.items():
        f.write(f"{metric_name}: {metric_value:.4f}\n")

    f.write("\n")
