import os
import json
import math
import torch
from sklearn.metrics import balanced_accuracy_score
from torch.utils.data import DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence

polish_to_english = {
    "Głowa lewą ręką": "Punch to the head with left hand",
    "Głowa prawą ręką": "Punch to the head with right hand",
    "Korpus lewą ręką": "Punch to the body with left hand",
    "Korpus prawą ręką": "Punch to the body with right hand",
    "Blok lewą ręką": "Block with left hand",
    "Blok prawą ręką": "Block with right hand",
    "Chybienie lewą ręką": "Missed punch with left hand",
    "Chybienie prawą ręką": "Missed punch with right hand"
}

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

num_keypoints = 17
input_dim = 4 * num_keypoints
num_classes = len(label2idx)

class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class TransformerClassifier(torch.nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers, hidden_dim, num_classes, dropout=0.3):
        super(TransformerClassifier, self).__init__()
        self.input_proj = torch.nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                         dim_feedforward=hidden_dim, dropout=dropout)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.dropout = torch.nn.Dropout(dropout)
        self.fc = torch.nn.Linear(d_model, num_classes)
    def forward(self, x, lengths):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)
        device = x.device
        lengths = lengths.to(device)
        max_len = x.size(0)
        mask = (torch.arange(max_len, device=device)[None, :] >= lengths[:, None])
        x = self.transformer_encoder(x, src_key_padding_mask=mask)
        x = x.mean(dim=0)
        x = self.dropout(x)
        logits = self.fc(x)
        return logits

class MultiFolderBoxingDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, num_keypoints=17, augment=False, augmentation_std=0.01):
        self.num_keypoints = num_keypoints
        self.entries = []
        self.augment = augment
        self.augmentation_std = augmentation_std
        for folder in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder)
            if os.path.isdir(folder_path):
                annotations_path = os.path.join(folder_path, "annotations.json")
                keypoints_path = os.path.join(folder_path, "data", "keypoints.json")
                if os.path.exists(annotations_path) and os.path.exists(keypoints_path):
                    with open(annotations_path, 'r', encoding='utf-8') as f:
                        annotations = json.load(f)
                    with open(keypoints_path, 'r', encoding='utf-8') as f:
                        keypoints_data = json.load(f)
                    frame_to_keypoints = {}
                    for item in keypoints_data:
                        frame = item.get('frame')
                        kp = item.get('keypoints')
                        if kp is None:
                            kp = {"1": [[0, 0]] * self.num_keypoints, "2": [[0, 0]] * self.num_keypoints}
                        frame_to_keypoints[frame] = kp
                    for ann in annotations:
                        tracks = ann.get("tracks", [])
                        for track in tracks:
                            label_polish = track.get("label", "unknown")
                            label_english = polish_to_english.get(label_polish, label_polish)
                            if label_english not in label2idx:
                                continue
                            shape_frames = [shape.get("frame") for shape in track.get("shapes", []) if "frame" in shape]
                            if not shape_frames:
                                continue
                            start_frame = min(shape_frames)
                            end_frame = max(shape_frames)
                            seq = []
                            for frame in range(start_frame, end_frame + 1):
                                kp = frame_to_keypoints.get(frame)
                                if kp is None:
                                    kp1 = [[0, 0]] * self.num_keypoints
                                    kp2 = [[0, 0]] * self.num_keypoints
                                else:
                                    kp1 = kp.get("1") if kp.get("1") is not None else [[0, 0]] * self.num_keypoints
                                    kp2 = kp.get("2") if kp.get("2") is not None else [[0, 0]] * self.num_keypoints
                                feat = [coord for point in kp1 for coord in point] + \
                                       [coord for point in kp2 for coord in point]
                                seq.append(feat)
                            sequence_tensor = torch.tensor(seq, dtype=torch.float32)
                            label_index = label2idx[label_english]
                            self.entries.append((sequence_tensor, label_index))
                else:
                    print(f"Skipping folder {folder_path}: required files not found.")
    def __len__(self):
        return len(self.entries)
    def __getitem__(self, idx):
        seq, label = self.entries[idx]
        if self.augment:
            seq = seq + torch.randn_like(seq) * self.augmentation_std
        return seq, label

def collate_fn(batch):
    sequences, labels = zip(*batch)
    lengths = torch.tensor([seq.shape[0] for seq in sequences], dtype=torch.int64)
    padded_sequences = pad_sequence(sequences, batch_first=True)
    labels = torch.tensor(labels, dtype=torch.long)
    return padded_sequences, lengths, labels

root_dir = r"C:\Users\willi\PycharmProjects\segment-anything-2-real-time\Olympic Boxing Punch Classification Video Dataset"
full_dataset = MultiFolderBoxingDataset(root_dir, num_keypoints=num_keypoints, augment=False)
print(f"Full dataset contains {len(full_dataset)} samples.")

test_dataset = full_dataset
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
best_model = TransformerClassifier(
    input_dim=input_dim,
    d_model=128,
    nhead=8,
    num_encoder_layers=3,
    hidden_dim=256,
    num_classes=num_classes,
    dropout=0.3
)
model_path = "best_boxing_action_transformer.pth"
best_model.load_state_dict(torch.load(model_path, map_location=device))
best_model.to(device)
best_model.eval()

total_samples = 0
correct_overall = 0
correct_punch = 0
punch_total = 0

def is_punch(label):
    return label < 9

with torch.no_grad():
    for sequences, lengths, labels in test_loader:
        sequences = sequences.to(device)
        labels = labels.to(device)
        outputs = best_model(sequences, lengths)
        _, predicted = torch.max(outputs, 1)
        total_samples += labels.size(0)
        correct_overall += (predicted == labels).sum().item()
        for gt, pred in zip(labels, predicted):
            if is_punch(gt.item()):
                punch_total += 1
                if is_punch(pred.item()):
                    correct_punch += 1

overall_accuracy = correct_overall / total_samples * 100
if punch_total > 0:
    punch_accuracy = correct_punch / punch_total * 100
else:
    punch_accuracy = 0

all_labels = []
all_predictions = []

with torch.no_grad():
    for sequences, lengths, labels in test_loader:
        sequences = sequences.to(device)
        labels = labels.to(device)
        outputs = best_model(sequences, lengths)
        _, predicted = torch.max(outputs, 1)
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

balanced_acc = balanced_accuracy_score(all_labels, all_predictions) * 100
print(f"Balanced Accuracy Label: {balanced_acc:.2f}%")

print(f"Overall Classification Accuracy: {overall_accuracy:.2f}%")
print(f"Punch Detection Accuracy (punch/block cases): {punch_accuracy:.2f}%")
