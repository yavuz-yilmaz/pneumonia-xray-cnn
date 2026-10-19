"""Predict with a downloaded DINOv3 checkpoint, without fetching pretrained weights."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import LABEL_TO_ID  # noqa: E402
from src.training.dinov3_model import DINOv3Classifier  # noqa: E402


def main() -> None:
    """Use only a checkpoint produced by this training workflow."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    args = parser.parse_args()
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != "dinov3_binary_v1" or checkpoint["label_mapping"] != LABEL_TO_ID:
        raise ValueError("Expected a DINOv3 binary checkpoint from this workflow")
    model = DINOv3Classifier.from_config(checkpoint["backbone_config"])
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    size = checkpoint["image_size"]
    normalization = checkpoint["normalization"]
    transform = transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(normalization["mean"], normalization["std"]),
        ]
    )
    with Image.open(args.image) as source:
        inputs = transform(ImageOps.exif_transpose(source).convert("RGB")).unsqueeze(0)
    with torch.inference_mode():
        logits = model(inputs)
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Nonfinite inference logits")
        probability = float(logits.softmax(1)[0, 1])
    threshold = checkpoint["decision_threshold"]
    print(
        json.dumps(
            {
                "predicted_label": "PNEUMONIA" if probability >= threshold else "NORMAL",
                "pneumonia_probability": probability,
                "normal_probability": 1 - probability,
                "decision_threshold": threshold,
            }
        )
    )


if __name__ == "__main__":
    main()
