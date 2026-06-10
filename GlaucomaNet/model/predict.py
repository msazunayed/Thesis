from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from model import GlaucomaCNN

SAVED_DIR = Path(__file__).parent / "saved"
WEIGHTS   = SAVED_DIR / "best_model.pth"
META_FILE = SAVED_DIR / "model_meta.json"


def build_transform(size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


class GlaucomaPredictor:
    # Loads the trained model once and exposes predict()

    def __init__(self):
        with open(META_FILE) as f:
            self.meta = json.load(f)

        self.class_names = self.meta["class_names"]
        self.input_size  = self.meta.get("input_size", 128)
        self.transform   = build_transform(self.input_size)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model  = GlaucomaCNN(num_classes=self.meta["num_classes"])
        self.model.load_state_dict(
            torch.load(WEIGHTS, map_location=self.device, weights_only=True)
        )
        self.model.to(self.device).eval()

    def predict(self, image: Image.Image) -> dict:
        if image.mode != "RGB":
            image = image.convert("RGB")

        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            probs = F.softmax(self.model(tensor), dim=1)[0]

        predicted_idx   = probs.argmax().item()
        predicted_label = self.class_names[predicted_idx]
        confidence      = round(probs[predicted_idx].item(), 4)

        if predicted_label == "GON+":
            interpretation = (
                f"Glaucomatous optic neuropathy detected (confidence: {confidence:.1%}). "
                "Please consult an ophthalmologist."
            )
        else:
            interpretation = (
                f"No glaucomatous optic neuropathy detected (confidence: {confidence:.1%})."
            )

        return {
            "label":         predicted_label,
            "confidence":    confidence,
            "probabilities": {
                cls: round(probs[i].item(), 4)
                for i, cls in enumerate(self.class_names)
            },
            "interpretation": interpretation,
        }


_predictor: GlaucomaPredictor | None = None


def get_predictor() -> GlaucomaPredictor:
    global _predictor
    if _predictor is None:
        _predictor = GlaucomaPredictor()
    return _predictor


def predict_image(image: Image.Image) -> dict:
    return get_predictor().predict(image)
