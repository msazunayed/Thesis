# Train GlaucomaCNN on the HYGD Glaucoma Dataset.
#
# Usage (run from the project root):
#     python model/train.py
#
# Expected dataset layout — place the unzipped HYGD folder inside data/:
#     data/
#         Images/      <- all .jpg fundus images
#         Labels.csv   <- Image Name, Patient, Label, Quality Score

import os, sys, csv, json, copy, time, random, shutil, argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

ROOT = Path(__file__).parent.parent

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# Inline model so train.py can be run standalone
class GlaucomaCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,  32,  3, padding=1), nn.BatchNorm2d(32),  nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64,  3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(128,256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True), nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512), nn.ReLU(True), nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class GlaucomaDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples   = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def load_labels(labels_csv: Path):
    img2label   = {}
    img2patient = {}
    with open(labels_csv, newline="") as f:
        for row in csv.DictReader(f):
            name    = row["Image Name"].strip()
            img2label[name]   = row["Label"].strip()
            img2patient[name] = row["Patient"].strip()
    return img2label, img2patient


def prepare_splits(images_dir: Path, labels_csv: Path, out_root: Path):
    # Patient-level 70/15/15 split to prevent data leakage
    img2label, img2patient = load_labels(labels_csv)

    patient2imgs = defaultdict(list)
    for img_name, patient in img2patient.items():
        patient2imgs[patient].append(img_name)

    patients = sorted(patient2imgs.keys())
    random.shuffle(patients)
    n         = len(patients)
    train_end = int(n * 0.70)
    val_end   = int(n * 0.85)

    split_map = {}
    for i, p in enumerate(patients):
        split_map[p] = "train" if i < train_end else ("val" if i < val_end else "test")

    class_names = ["GON-", "GON+"]
    for split in ("train", "val", "test"):
        for cls in class_names:
            (out_root / split / cls).mkdir(parents=True, exist_ok=True)

    counts = defaultdict(lambda: defaultdict(int))
    for img_name, label in img2label.items():
        patient = img2patient[img_name]
        split   = split_map[patient]
        src     = images_dir / img_name
        if not src.exists():
            continue
        dest = out_root / split / label / img_name
        if not dest.exists():
            shutil.copy2(src, dest)
        counts[split][label] += 1

    for split in ("train", "val", "test"):
        total = sum(counts[split].values())
        print(f"  {split:5s}: {total:4d} images  (GON+ {counts[split]['GON+']}, GON- {counts[split]['GON-']})")

    return class_names


def train(data_dir: Path, save_dir: Path, epochs: int, lr: float, batch_size: int,
          img_size: int = 128, class_names=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Image size: {img_size}×{img_size}")

    if class_names is None:
        class_names = sorted([d.name for d in (data_dir / "train").iterdir() if d.is_dir()])
    cls2idx = {c: i for i, c in enumerate(class_names)}

    def make_samples(split):
        samples = []
        for cls in class_names:
            for p in sorted((data_dir / split / cls).glob("*.jpg")):
                samples.append((p, cls2idx[cls]))
        return samples

    train_samples = make_samples("train")
    val_samples   = make_samples("val")

    # Weighted sampler to handle class imbalance (548 GON+ vs 199 GON-)
    class_counts  = [sum(1 for _, l in train_samples if l == i) for i in range(len(class_names))]
    class_weights = [1.0 / c if c > 0 else 0.0 for c in class_counts]
    sample_weights = torch.tensor([class_weights[l] for _, l in train_samples])
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights))

    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds = GlaucomaDataset(train_samples, train_tf)
    val_ds   = GlaucomaDataset(val_samples,   val_tf)
    train_dl = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,   num_workers=0)
    print(f"Classes: {class_names}  |  Train: {len(train_ds)}  |  Val: {len(val_ds)}")

    total   = sum(class_counts)
    weights = torch.tensor([total / (len(class_names) * c) for c in class_counts],
                           dtype=torch.float).to(device)

    model     = GlaucomaCNN(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    best_weights = copy.deepcopy(model.state_dict())
    history      = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        model.train()
        train_loss, train_correct = 0.0, 0
        for X, y in train_dl:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(X)
            loss   = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss    += loss.item() * X.size(0)
            train_correct += (logits.argmax(1) == y).sum().item()

        train_loss /= len(train_ds)
        train_acc   = train_correct / len(train_ds)

        model.eval()
        val_loss, val_correct = 0.0, 0
        with torch.no_grad():
            for X, y in val_dl:
                X, y   = X.to(device), y.to(device)
                logits  = model(X)
                loss    = criterion(logits, y)
                val_loss    += loss.item() * X.size(0)
                val_correct += (logits.argmax(1) == y).sum().item()

        val_loss /= len(val_ds)
        val_acc   = val_correct / len(val_ds)
        scheduler.step()

        history["train_loss"].append(round(train_loss, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["train_acc"].append(round(train_acc, 4))
        history["val_acc"].append(round(val_acc, 4))

        elapsed = time.time() - t0
        print(f"Epoch {epoch:3d}/{epochs}  train_acc={train_acc:.4f}  val_acc={val_acc:.4f}  ({elapsed:.0f}s)")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(best_weights, save_dir / "best_model.pth")
            print(f"  ✓ New best saved (val_acc={best_val_acc:.4f})")

    meta = {
        "class_names":  class_names,
        "num_classes":  len(class_names),
        "input_size":   img_size,
        "best_val_acc": round(best_val_acc, 4),
        "epochs":       epochs,
        "history":      history,
        "architecture": "GlaucomaCNN",
        "dataset":      "HYGD",
    }
    with open(save_dir / "model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n✓ Training complete  |  Best val acc: {best_val_acc:.4f}")
    print(f"  Saved to: {save_dir}/")
    print("  Next step: python model/test.py")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train GlaucomaCNN on the HYGD dataset")
    p.add_argument("--images_dir",   default=str(ROOT / "data" / "Images"))
    p.add_argument("--labels_csv",   default=str(ROOT / "data" / "Labels.csv"))
    p.add_argument("--prepared_dir", default=str(ROOT / "data_prepared"))
    p.add_argument("--save_dir",     default=str(ROOT / "model" / "saved"))
    p.add_argument("--epochs",   type=int,   default=30)
    p.add_argument("--lr",       type=float, default=3e-4)
    p.add_argument("--batch",    type=int,   default=32)
    p.add_argument("--img_size", type=int,   default=128)
    args = p.parse_args()

    print("── Preparing dataset splits ──")
    class_names = prepare_splits(
        Path(args.images_dir),
        Path(args.labels_csv),
        Path(args.prepared_dir),
    )
    print("── Training ──")
    train(
        Path(args.prepared_dir),
        Path(args.save_dir),
        args.epochs,
        args.lr,
        args.batch,
        args.img_size,
        class_names,
    )
