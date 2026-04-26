from datetime import datetime
import time
import argparse

import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

parser = argparse.ArgumentParser()
parser.add_argument("--data_name", type=str, required=True)
parser.add_argument("--result_txt_path", type=str, required=True)
parser.add_argument("--data_name", type=str, required=True)
parser.add_argument("--test_path", type=str, required=True)

parser.add_argument("--num_train_epochs", type=str, required=True)
parser.add_argument("--batch size", type=str, required=True)
args = parser.parse_args()

data_path = args.data_name
result_txt_path = args.result_txt_path
test_path = args.test_path

num_train_epochs=args.num_train_epochs
batch size = args.batch size

start_time = time.time()
start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

df = pd.read_excel(data_path)

if df.isnull().any().any():
    df = df.dropna()
dataset = Dataset.from_pandas(df)

test_df = pd.read_excel(test_path)
test_dataset = Dataset.from_pandas(test_df[['Sentence', 'label']])

tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path, num_labels=2)

def preprocess_function(examples):
    tokenized_inputs = tokenizer(examples['sentence'], truncation=True, padding='max_length', max_length=512)
    tokenized_inputs['labels'] = examples['label'] 
    return tokenized_inputs

dataset = dataset.map(preprocess_function, batched=True)

train_data = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = train_data['train']
val_dataset = train_data['test']

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    accuracy = accuracy_score(labels, preds)

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, preds, average='macro'
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        labels, preds, average='weighted'
    )

    return {
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
    }

training_args = TrainingArguments(
    output_dir=f'./results',       
    num_train_epochs=num_train_epochs,           
    per_device_train_batch_size=batch size, 
    per_device_eval_batch_size=batch size,  
    warmup_ratio=0.1,  
    weight_decay=0.01
    logging_dir=f'./logs',   
    logging_steps=50,
    eval_strategy="epoch",  
    save_strategy="epoch",  
    save_total_limit=2,
    load_best_model_at_end=True, 
    metric_for_best_model="eval_accuracy", 
    greater_is_better=True, 
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


# trainer.save_model(f'./model')
# tokenizer.save_pretrained(f'./model') 

def preprocess_test(examples):
    tokenized_inputs = tokenizer(examples['Sentence'], truncation=True, padding='max_length', max_length=512)
    return tokenized_inputs

test_dataset = test_dataset.map(preprocess_test, batched=True)

predictions = trainer.predict(test_dataset)
predicted_labels = np.argmax(predictions.predictions, axis=1)

true_labels = test_df['label'].values
eval_pred = (predictions.predictions, true_labels)
metrics = compute_metrics(eval_pred)
print(data_path)
for metric_name, metric_value in metrics.items():
    print(f"{metric_name}: {metric_value:.4f}")

training_params = {
    "data_path": data_path,
    "model_path": model_path,
    "num_train_epochs": num_train_epochs,
    "per_device_train_batch_size": batch_size,
    "per_device_eval_batch_size": batch_size,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "max_length": 512
}
end_time = time.time()
end_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

with open(result_txt_path, "a", encoding="utf-8") as f:

    f.write(f"start_datetime: {start_datetime}\n")
    f.write(f"end_datetime: {end_datetime}\n")

    # f.write("========== params ==========\n")
    # for k, v in training_params.items():
    #     f.write(f"{k}: {v}\n")

    f.write("\n========== data_info ==========\n")
    f.write(f"len(df): {len(df)}\n")
    f.write(f"len(train_dataset): {len(train_dataset)}\n")
    f.write(f"len(val_dataset): {len(val_dataset)}\n")
    f.write(f"len(test_dataset): {len(test_dataset)}\n")

    f.write("\n========== test result ==========\n")
    for metric_name, metric_value in metrics.items():
        f.write(f"{metric_name}: {metric_value:.4f}\n")




