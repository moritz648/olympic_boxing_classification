import os
import json
import math
import cv2
import torch

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
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=hidden_dim, dropout=dropout
        )
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

main_folder = r"C:\Users\willi\PycharmProjects\segment-anything-2-real-time\Olympic Boxing Punch Classification Video Dataset"

with open("data_split_indices.json", "r") as f:
    split_indices = json.load(f)
test_indices = split_indices["test"]

subfolders_all = sorted([
    os.path.join(main_folder, d)
    for d in os.listdir(main_folder)
    if os.path.isdir(os.path.join(main_folder, d))
])
test_folders = [subfolders_all[i] for i in test_indices if i < len(subfolders_all)]
print(f"Evaluating on {len(test_folders)} test folders.")
global_TP = 0
global_TN = 0
global_FP = 0
global_FN = 0
global_total_frames = 0

for folder in test_folders:
    print(f"\nProcessing folder: {folder}")
    data_folder = os.path.join(folder, "data")

    video_files = [f for f in os.listdir(data_folder) if f.lower().endswith('.mp4')]
    if not video_files:
        print(f"  No video file found in {data_folder}, skipping folder.")
        continue
    video_path = os.path.join(data_folder, video_files[0])

    annotations_path = os.path.join(folder, "annotations.json")
    keypoints_path = os.path.join(data_folder, "keypoints.json")

    if not os.path.exists(annotations_path) or not os.path.exists(keypoints_path):
        print(f"  Missing annotations or keypoints in folder {folder}, skipping folder.")
        continue

    with open(annotations_path, 'r', encoding='utf-8') as f:
        annotations = json.load(f)
    with open(keypoints_path, 'r', encoding='utf-8') as f:
        keypoints_data = json.load(f)

    frame_to_keypoints = {}
    for item in keypoints_data:
        frame = item.get('frame')
        kp = item.get('keypoints')
        if kp is None:
            kp = {"1": [[0, 0]] * num_keypoints, "2": [[0, 0]] * num_keypoints}
        frame_to_keypoints[frame] = kp

    predictions = []
    ground_truth = {}

    for ann in annotations:
        tracks = ann.get("tracks", [])
        for track in tracks:
            label_polish = track.get("label", "unknown")
            label_english = polish_to_english.get(label_polish, label_polish)
            if label_english not in label2idx:
                continue

            shape_frames = []
            for shape in track.get("shapes", []):
                frame_num = shape.get("frame")
                points = shape.get("points", [])
                if frame_num is not None and len(points) == 4:
                    shape_frames.append(frame_num)
                    x1, y1, x2, y2 = map(int, points)
                    if frame_num not in ground_truth:
                        ground_truth[frame_num] = []
                    ground_truth[frame_num].append(((x1, y1, x2, y2), label_english))
            if not shape_frames:
                continue

            start_frame = min(shape_frames)
            end_frame = max(shape_frames)

            seq = []
            for frame in range(start_frame, end_frame + 1):
                kp = frame_to_keypoints.get(frame)
                if kp is None:
                    kp1 = [[0, 0]] * num_keypoints
                    kp2 = [[0, 0]] * num_keypoints
                else:
                    kp1 = kp.get("1") if kp.get("1") is not None else [[0, 0]] * num_keypoints
                    kp2 = kp.get("2") if kp.get("2") is not None else [[0, 0]] * num_keypoints
                feat = [coord for point in kp1 for coord in point] + [coord for point in kp2 for coord in point]
                seq.append(feat)

            sequence_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
            length_tensor = torch.tensor([len(seq)], dtype=torch.int64)

            with torch.no_grad():
                output = best_model(sequence_tensor.to(device), length_tensor.to(device))
                _, predicted_idx = torch.max(output, 1)
                predicted_label = idx2label[predicted_idx.item()]

            predictions.append((start_frame, end_frame, predicted_label))

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    y_true = [0] * total_frames
    y_pred = [0] * total_frames

    for frame in ground_truth:
        if 0 <= frame < total_frames:
            y_true[frame] = 1

    for (start_frame, end_frame, _) in predictions:
        for frame in range(start_frame, end_frame + 1):
            if 0 <= frame < total_frames:
                y_pred[frame] = 1

    TP = sum(1 for i in range(total_frames) if y_true[i] == 1 and y_pred[i] == 1)
    TN = sum(1 for i in range(total_frames) if y_true[i] == 0 and y_pred[i] == 0)
    FP = sum(1 for i in range(total_frames) if y_true[i] == 0 and y_pred[i] == 1)
    FN = sum(1 for i in range(total_frames) if y_true[i] == 1 and y_pred[i] == 0)

    print(f"  Frames: {total_frames}")
    print(f"  True Positives: {TP}, True Negatives: {TN}, False Positives: {FP}, False Negatives: {FN}")

    global_TP += TP
    global_TN += TN
    global_FP += FP
    global_FN += FN
    global_total_frames += total_frames

overall_recall_action = global_TP / (global_TP + global_FN) if (global_TP + global_FN) > 0 else 0
overall_recall_no_action = global_TN / (global_TN + global_FP) if (global_TN + global_FP) > 0 else 0
overall_balanced_accuracy = (overall_recall_action + overall_recall_no_action) / 2
overall_accuracy = (global_TP + global_TN) / global_total_frames

print("\nOverall evaluation on test set:")
print(f"Total frames: {global_total_frames}")
print(f"Overall True Positives: {global_TP}")
print(f"Overall True Negatives: {global_TN}")
print(f"Overall False Positives: {global_FP}")
print(f"Overall False Negatives: {global_FN}")
print(f"Overall Recall for action: {overall_recall_action:.4f}")
print(f"Overall Recall for no action: {overall_recall_no_action:.4f}")
print(f"Overall Balanced Accuracy: {overall_balanced_accuracy:.4f}")
print(f"Overall Accuracy: {overall_accuracy:.4f}")
