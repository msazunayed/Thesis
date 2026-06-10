# Evaluate the saved GlaucomaCNN on the held-out test set.
#
# Usage (run from the project root after training):
#     python model/test.py

import sys, json
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image

ROOT     = Path(__file__).parent.parent
SAVED    = Path(__file__).parent / "saved"
PREPARED = ROOT / "data_prepared"

sys.path.insert(0, str(Path(__file__).parent))
from model import GlaucomaCNN


class SimpleDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples   = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


def test():
    meta_path = SAVED / "model_meta.json"
    if not meta_path.exists():
        print("✗ model_meta.json not found. Run train.py first.")
        return

    with open(meta_path) as f:
        meta = json.load(f)

    class_names = meta["class_names"]
    img_size    = meta.get("input_size", 128)
    n_classes   = len(class_names)
    cls2idx     = {c: i for i, c in enumerate(class_names)}

    test_dir = PREPARED / "test"
    if not test_dir.exists():
        print("✗ Test set not found. Re-run train.py to regenerate splits.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = GlaucomaCNN(num_classes=n_classes)
    model.load_state_dict(
        torch.load(SAVED / "best_model.pth", map_location=device, weights_only=True)
    )
    model.to(device).eval()

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    samples = []
    for cls in class_names:
        for p in sorted((test_dir / cls).glob("*.jpg")):
            samples.append((p, cls2idx[cls]))

    test_ds = SimpleDataset(samples, transform)
    test_dl = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)
    print(f"Test set: {len(test_ds)} images  |  Classes: {class_names}")

    confusion  = [[0] * n_classes for _ in range(n_classes)]
    all_probs  = []
    all_labels = []
    correct    = 0

    with torch.no_grad():
        for X, y in test_dl:
            X, y   = X.to(device), y.to(device)
            logits  = model(X)
            probs   = F.softmax(logits, dim=1)
            preds   = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            for actual, pred in zip(y.tolist(), preds.tolist()):
                confusion[actual][pred] += 1
            all_probs.extend(probs[:, -1].cpu().tolist())
            all_labels.extend(y.cpu().tolist())

    total    = len(test_ds)
    accuracy = correct / total * 100

    print(f"\n{'─' * 44}")
    print(f"  Test Accuracy : {accuracy:.2f}%  ({correct}/{total})")
    print(f"{'─' * 44}")

    print("\n  Per-class accuracy:")
    for i, cls in enumerate(class_names):
        row_total = sum(confusion[i])
        cls_acc   = confusion[i][i] / row_total * 100 if row_total > 0 else 0
        print(f"    {cls:<6}  {cls_acc:.2f}%  ({confusion[i][i]}/{row_total})")

    if n_classes == 2:
        gon_idx = cls2idx.get("GON+", 1)
        neg_idx = 1 - gon_idx
        TP = confusion[gon_idx][gon_idx]
        FN = confusion[gon_idx][neg_idx]
        FP = confusion[neg_idx][gon_idx]
        TN = confusion[neg_idx][neg_idx]
        sensitivity = TP / (TP + FN) * 100 if (TP + FN) > 0 else 0
        specificity = TN / (TN + FP) * 100 if (TN + FP) > 0 else 0
        print(f"\n  Sensitivity (GON+ recall) : {sensitivity:.2f}%")
        print(f"  Specificity (GON- recall) : {specificity:.2f}%")

        # AUC via trapezoid rule
        pairs   = sorted(zip(all_probs, all_labels), reverse=True)
        tp = fp = 0
        P       = sum(l == gon_idx for l in all_labels)
        N       = total - P
        roc_pts = [(0.0, 0.0)]
        for _, label in pairs:
            if label == gon_idx:
                tp += 1
            else:
                fp += 1
            roc_pts.append((fp / N if N else 0, tp / P if P else 0))
        auc = sum(
            (roc_pts[i][0] - roc_pts[i-1][0]) * (roc_pts[i][1] + roc_pts[i-1][1]) / 2
            for i in range(1, len(roc_pts))
        )
        print(f"  AUC                       : {auc:.4f}")

    print("\n  Confusion matrix (rows=actual, cols=predicted):")
    pad    = max(len(c) for c in class_names) + 2
    header = " " * (pad + 9) + "  ".join(f"pred_{c}" for c in class_names)
    print(f"  {header}")
    for i, cls in enumerate(class_names):
        row = "  ".join(
            f"{confusion[i][j]:>{len('pred_' + class_names[j])}}"
            for j in range(n_classes)
        )
        print(f"  actual_{cls:<{pad}}{row}")

    print(f"\n  Val acc (training) : {meta['best_val_acc'] * 100:.2f}%")
    print(f"  Test acc           : {accuracy:.2f}%")
    diff = accuracy - meta["best_val_acc"] * 100
    if abs(diff) < 5:
        print("  ✓ Val and test are close — model generalizes well")
    elif diff < -5:
        print("  ⚠ Test is below val — possible overfitting")
    else:
        print("  ✓ Test exceeds val — solid generalization")


if __name__ == "__main__":
    test()
