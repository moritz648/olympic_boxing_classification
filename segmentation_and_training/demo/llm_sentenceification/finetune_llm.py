import os
import math
import pandas as pd
import torch
from collections import Counter
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          Trainer, TrainingArguments, DataCollatorWithPadding)
from datasets import load_dataset
from sklearn.metrics import accuracy_score

label2idx = {
    "Punch to the head with left hand": 0,
    "Punch to the head with right hand": 1,
    "Punch to the body with left hand": 2,
    "Punch to the body with right hand": 3,
    "Block with left hand": 4,
    "Block with right hand": 5,
    "Missed punch with left hand": 6,
    "Missed punch with right hand": 7
}
idx2label = {v: k for k, v in label2idx.items()}
num_classes = len(label2idx)

train_df = pd.read_csv("train_sentencified_full.csv")
train_labels = train_df["label"].tolist()
counter = Counter(train_labels)
total = len(train_labels)

weights = [total / (num_classes * counter[i]) for i in range(num_classes)]
class_weights = torch.tensor(weights, dtype=torch.float32)
print("Class weights:", class_weights)

class WeightedLossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights.to(model.device))
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}

model_name = "roberta-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_classes)

data_files = {"train": "train_sentencified_full.csv", "validation": "val_sentencified_full.csv"}
hf_datasets = load_dataset("csv", data_files=data_files)

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)

tokenized_train = hf_datasets["train"].map(tokenize_function, batched=True, remove_columns=["text"])
tokenized_val = hf_datasets["validation"].map(tokenize_function, batched=True, remove_columns=["text"])

tokenized_train.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
tokenized_val.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

training_args = TrainingArguments(
    output_dir="./seq_classification_full_results",
    overwrite_output_dir=True,
    eval_strategy="steps",
    eval_steps=500,
    save_steps=500,
    learning_rate=1e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=10,
    weight_decay=0.05,
    warmup_steps=500,
    logging_steps=100,
    gradient_accumulation_steps=4,
)

trainer = WeightedLossTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

model.save_pretrained("./fine_tuned_seq_classification_full")
tokenizer.save_pretrained("./fine_tuned_seq_classification_full")

print("Fine-tuning complete. Model saved in './fine_tuned_seq_classification_full'.")
