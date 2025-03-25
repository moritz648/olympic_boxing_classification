from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import balanced_accuracy_score
import torch

idx2label = {
    0: "Punch to the head with left hand",
    1: "Punch to the head with right hand",
    2: "Punch to the body with left hand",
    3: "Punch to the body with right hand",
    4: "Block with left hand",
    5: "Block with right hand",
    6: "Missed punch with left hand",
    7: "Missed punch with right hand"
}

label2idx = {v: k for k, v in idx2label.items()}

model_path = "./fine_tuned_seq_classification_full"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

true_labels = []
predicted_labels = []

print("=== Testing on val_sentencified.txt ===")
with open("val_sentencified.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

for line in tqdm(lines):
    line = line.strip()
    if not line:
        continue
    if "Action:" in line:
        prompt_part, true_label_part = line.split("Action:", 1)
        prompt = prompt_part.strip() + " Action:"
        true_label = true_label_part.strip().rstrip(".")
    else:
        continue

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = model(**inputs)
    predicted_index = outputs.logits.argmax(dim=-1).item()
    predicted_label = idx2label[predicted_index]

    if true_label in label2idx:
        true_labels.append(label2idx[true_label])
        predicted_labels.append(predicted_index)

# Calculate balanced accuracy
balanced_acc = balanced_accuracy_score(true_labels, predicted_labels)
print(f"\nBalanced Accuracy: {balanced_acc:.4f}")
