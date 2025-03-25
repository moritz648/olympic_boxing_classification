import os
import json
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm
import numpy as np

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

class VideoPunchDataset(Dataset):
    def __init__(self, root_dir, transform=None, use_full_frame=True):
        self.samples = []
        self.transform = transform
        self.use_full_frame = use_full_frame

        for folder in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder)
            if os.path.isdir(folder_path):
                video_path = None
                data_folder = os.path.join(folder_path, "data")
                if os.path.isdir(data_folder):
                    for file in os.listdir(data_folder):
                        if 'mp4' in file:
                            video_path = os.path.join(data_folder, file)
                            break
                annotations_path = os.path.join(folder_path, "annotations.json")
                if video_path is not None and os.path.exists(annotations_path):
                    with open(annotations_path, 'r', encoding='utf-8') as f:
                        annotations = json.load(f)
                    for ann in annotations:
                        tracks = ann.get("tracks", [])
                        for track in tracks:
                            label_polish = track.get("label", "unknown")
                            label_english = polish_to_english.get(label_polish, label_polish)
                            if label_english not in label2idx:
                                continue
                            class_idx = label2idx[label_english]
                            shapes = track.get("shapes", [])
                            for shape in shapes:
                                if "frame" not in shape or "points" not in shape:
                                    continue
                                frame_num = int(shape["frame"])
                                points = shape["points"]
                                if len(points) != 4:
                                    continue
                                x1, y1, x2, y2 = points
                                bbox = [int(x1), int(y1), int(x2), int(y2)]
                                self.samples.append({
                                    "video_path": video_path,
                                    "frame": frame_num,
                                    "bbox": bbox,
                                    "label": class_idx
                                })
                else:
                    print(f"Skipping folder {folder_path}: missing video or annotations.json")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        video_path = sample["video_path"]
        frame_num = sample["frame"]
        bbox = sample["bbox"]
        label = sample["label"]

        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise RuntimeError(f"Failed to read frame {frame_num} from {video_path}")

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if self.use_full_frame:
            image = Image.fromarray(frame)
        else:
            x1, y1, x2, y2 = bbox
            cropped = frame[y1:y2, x1:x2]
            if cropped.size == 0:
                cropped = frame
            image = Image.fromarray(cropped)

        if self.transform:
            image = self.transform(image)

        return image, label

if __name__ == "__main__":
    root_dir = r"C:\Users\willi\PycharmProjects\segment-anything-2-real-time\Olympic Boxing Punch Classification Video Dataset"

    data_transforms = transforms.Compose([
        transforms.Resize((180, 180)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    dataset = VideoPunchDataset(root_dir, transform=data_transforms, use_full_frame=True)
    print(f"Loaded {len(dataset)} samples from the dataset.")

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    class_counts = np.zeros(len(label2idx))
    for _, label in train_dataset:
        class_counts[label] += 1
    print("Class counts:", class_counts)
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / np.sum(class_weights) * len(label2idx)
    class_weights = torch.FloatTensor(class_weights).to('cuda' if torch.cuda.is_available() else 'cpu')
    print("Class weights:", class_weights)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(label2idx))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 20

    checkpoint_path = "checkpoint.pth"
    start_epoch = 0

    if os.path.exists(checkpoint_path):
        print("Loading checkpoint...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")

    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False)
        for images, labels in train_bar:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            train_bar.set_postfix(loss=loss.item())
        epoch_loss = running_loss / len(train_dataset)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss,
        }, checkpoint_path)
        print(f"Checkpoint saved at epoch {epoch + 1}")

    model.eval()
    correct = 0
    total = 0
    eval_bar = tqdm(test_loader, desc="Evaluating", leave=False)
    with torch.no_grad():
        for images, labels in eval_bar:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")

    torch.save(model.state_dict(), "cnn_punch_detector.pth")
    print("Final model saved as cnn_punch_detector.pth")
