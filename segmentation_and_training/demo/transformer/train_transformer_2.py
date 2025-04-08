import os
import json
import math
import random
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import ReduceLROnPlateau

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

class MultiFolderBoxingDataset(Dataset):
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

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs, targets):
        ce_loss = self.ce_loss(inputs, targets)
        pt = torch.exp(-ce_loss)
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_factor = self.alpha[targets]
        else:
            alpha_factor = 1.0
        focal_loss = alpha_factor * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
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


class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers, hidden_dim, num_classes, dropout=0.3):
        super(TransformerClassifier, self).__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=hidden_dim, dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, num_classes)

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

def train_and_validate(model, train_loader, val_loader, num_epochs, learning_rate, criterion, patience=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

    best_val_loss = float('inf')
    best_model_state = None
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for sequences, lengths, labels in train_loader:
            sequences = sequences.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(sequences, lengths)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * sequences.size(0)
        train_loss = total_loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for sequences, lengths, labels in val_loader:
                sequences = sequences.to(device)
                labels = labels.to(device)
                outputs = model(sequences, lengths)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * sequences.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        val_loss /= len(val_loader.dataset)
        val_accuracy = correct / total * 100

        print(
            f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.2f}%")
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print("Early stopping triggered.")
            break

    model.load_state_dict(best_model_state)
    return model, best_val_loss


if __name__ == "__main__":
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)

    root_dir = "../../Olympic Boxing Punch Classification Video Dataset"
    num_keypoints = 17
    input_dim = 4 * num_keypoints

    dataset = MultiFolderBoxingDataset(root_dir, num_keypoints=num_keypoints, augment=True, augmentation_std=0.01)
    print(f"Loaded {len(dataset)} samples from multiple folders.")

    total_samples = len(dataset)
    train_val_size = int(0.8 * total_samples)
    test_size = total_samples - train_val_size
    generator = torch.Generator().manual_seed(seed)
    train_val_dataset, test_dataset = random_split(dataset, [train_val_size, test_size], generator=generator)

    train_size = int(0.8 * train_val_size)
    val_size = train_val_size - train_size
    train_dataset, val_dataset = random_split(train_val_dataset, [train_size, val_size], generator=generator)
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}, Test samples: {len(test_dataset)}")

    split_indices = {
        "train": train_dataset.indices,
        "val": val_dataset.indices,
        "test": test_dataset.indices
    }
    with open("data_split_indices.json", "w") as f:
        json.dump(split_indices, f)
    print("Saved train/val/test split indices to 'data_split_indices.json'.")

    batch_size = 32

    train_labels = [train_dataset[i][1] for i in range(len(train_dataset))]
    counter = Counter(train_labels)
    total_train = len(train_labels)
    num_classes = len(label2idx)
    class_weights_list = [total_train / (num_classes * counter[i]) for i in range(num_classes)]
    class_weights = torch.tensor(class_weights_list, dtype=torch.float32)
    print("Class weights:", class_weights)

    sample_weights = [class_weights[train_dataset[i][1]].item() for i in range(len(train_dataset))]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_dataset), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    best_lr = 0.001
    best_dropout = 0.3
    best_hidden_dim = 256
    best_num_encoder_layers = 3
    best_nhead = 8
    best_d_model = 128

    model = TransformerClassifier(
        input_dim=input_dim,
        d_model=best_d_model,
        nhead=best_nhead,
        num_encoder_layers=best_num_encoder_layers,
        hidden_dim=best_hidden_dim,
        num_classes=len(label2idx),
        dropout=best_dropout
    )

    num_epochs = 500
    criterion = FocalLoss(alpha=class_weights, gamma=2, reduction='mean')

    model, val_loss = train_and_validate(model, train_loader, val_loader, num_epochs, best_lr, criterion, patience=50)
    print(f"Best validation loss: {val_loss:.4f}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for sequences, lengths, labels in test_loader:
            sequences = sequences.to(device)
            labels = labels.to(device)
            outputs = model(sequences, lengths)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
    test_accuracy = 100 * correct / total
    print(f"Test Accuracy: {test_accuracy:.2f}%")

    try:
        from sklearn.metrics import confusion_matrix, classification_report

        cm = confusion_matrix(all_labels, all_preds)
        print("Confusion Matrix:")
        print(cm)
        print("Classification Report:")
        print(classification_report(all_labels, all_preds, target_names=list(label2idx.keys())))
    except ImportError:
        print("sklearn is not installed; skipping confusion matrix and classification report.")

    torch.save(model.state_dict(), "best_boxing_action_transformer.pth")
    print("Trained model saved to 'best_boxing_action_transformer.pth'.")
