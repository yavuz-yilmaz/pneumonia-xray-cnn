"""Pretrain opacity features on audited RSNA development patients only."""

from __future__ import annotations

import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.core.reproducibility import set_global_seed
from src.data.image_preparation import CheckpointImageTransform, prepare_image
from src.training.models import build_model
from src.training.train_clean import train_transform, validate


class OpacityDataset(Dataset):
    """Keep opacity labels distinct from the deployed pneumonia label mapping."""

    def __init__(self, records: list[dict], training: bool) -> None:
        self.records = records
        self.transform = train_transform(224) if training else CheckpointImageTransform(224)
        self.images = []
        for record in records:
            path = Path(record["filepath"])
            if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
                raise ValueError("RSNA source image changed")
            with Image.open(path) as image:
                self.images.append(prepare_image(image, 224, "clahe"))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return self.transform(self.images[index]), self.records[index]["label_id"]


def main() -> None:
    """Run one fixed feature-pretraining recipe; never load external test images."""
    directory = Path("data/processed/rsna_development_v1")
    output = Path("models/clean_v6/rsna_pretrained")
    if output.exists():
        raise FileExistsError("Pretraining run already exists")
    audit = json.loads((directory / "development_audit.json").read_bytes())
    content = (directory / "images.json").read_bytes()
    if audit["status"] != "passed" or audit["cross_split_components"]:
        raise ValueError("RSNA development audit has not passed")
    if hashlib.sha256(content).hexdigest() != audit["development_manifest_sha256"]:
        raise ValueError("Audited development manifest changed")
    records = json.loads(content)
    torch.set_num_threads(4)
    set_global_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    protocol = {
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": "resnet18",
        "initialization": "ImageNet",
        "seed": 42,
        "target": {"Normal": 0, "Lung Opacity": 1},
        "audit": audit,
        "epochs": 16,
        "patience": 5,
        "learning_rate": 0.0003,
        "preprocessing": "clahe",
        "size": 224,
        "batch_size": 24,
        "selection": "Minimum unweighted development validation CE",
        "transfer": "Discard opacity classifier; transfer backbone to pediatric training",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    output.mkdir(parents=True)
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print("Preparing audited RSNA images", flush=True)
    train = OpacityDataset([r for r in records if r["split"] == "train"], True)
    val = OpacityDataset([r for r in records if r["split"] == "val"], False)
    train_loader = DataLoader(
        train, batch_size=24, shuffle=True, generator=torch.Generator().manual_seed(42)
    )
    val_loader = DataLoader(val, batch_size=24)
    model = build_model("resnet18", 2, pretrained=True, freeze_backbone=True).to(device)
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [p for name, p in model.named_parameters() if not name.startswith("fc.")],
                "lr": 0.00006,
            },
            {"params": model.fc.parameters(), "lr": 0.0003},
        ],
        weight_decay=1e-4,
    )
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    history, best, stale = [], float("inf"), 0
    started = time.monotonic()
    for epoch in range(16):
        if epoch == 2:
            for parameter in model.parameters():
                parameter.requires_grad = True
        model.train()
        if epoch < 2:
            model.eval()
            model.fc.train()
        multiplier = 0.05 + 0.95 * (1 + math.cos(math.pi * max(0, epoch - 2) / 14)) / 2
        for group, lr in zip(optimizer.param_groups, [0.00006, 0.0003], strict=True):
            group["lr"] = lr * multiplier
        loss_sum = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss = nn.functional.cross_entropy(model(images), labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach()) * len(labels)
        labels, scores, val_loss, _ = validate(model, val_loader, device)
        entry = {
            "epoch": epoch + 1,
            "train_loss": loss_sum / len(train),
            "val_loss": val_loss,
            "elapsed_seconds": time.monotonic() - started,
        }
        history.append(entry)
        (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(json.dumps(entry), flush=True)
        if val_loss < best:
            best, stale = val_loss, 0
            torch.save(
                {
                    "model_name": "resnet18",
                    "feature_state_dict": {
                        k: v.detach().cpu()
                        for k, v in model.state_dict().items()
                        if not k.startswith("fc.")
                    },
                    "pretraining_protocol": protocol,
                    "epoch": epoch + 1,
                },
                output / "features.pt",
            )
            np.savez(output / "validation_predictions.npz", labels=labels, probabilities=scores)
        else:
            stale += 1
        if epoch >= 7 and stale >= 5:
            break
    (output / "result.json").write_text(
        json.dumps({"status": "complete", "epochs": len(history), "best_validation_loss": best}),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
