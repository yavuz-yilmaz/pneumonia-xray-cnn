"""Versioned full fine-tuning runs that never open the test manifest or test images."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms

from src.core.reproducibility import set_global_seed
from src.data.dataset import LABEL_TO_ID, ChestXRayDataset
from src.data.image_preparation import cache_prepared_images
from src.data.nih_dataset import NIH_LABEL_TO_ID, NIHReportDataset
from src.data.transforms import (
    IMAGENET_NORMALIZATION_MEAN,
    IMAGENET_NORMALIZATION_STD,
)
from src.evaluation.decision import binary_metrics, select_f1_threshold, select_threshold
from src.training.models import build_model, get_classifier_module


def verify_manifests(directory: Path) -> dict:
    """Require a successful audit and matching train/validation manifest hashes."""
    audit = json.loads((directory / "split_audit.json").read_text(encoding="utf-8"))
    if audit["cross_split_near_duplicates"] or any(
        any(values.values()) for values in audit["overlaps"].values()
    ):
        raise ValueError("Dataset audit reports leakage")
    for split in ("train", "val"):
        path = directory / f"{split}_manifest.csv"
        if hashlib.sha256(path.read_bytes()).hexdigest() != audit["manifest_sha256"][split]:
            raise ValueError(f"Manifest changed after audit: {split}")
        with path.open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                source_path = Path(row["filepath"])
                if hashlib.sha256(source_path.read_bytes()).hexdigest() != row["sha256"]:
                    raise ValueError(f"Image changed after audit: {source_path}")
    return audit


def train_transform(
    image_size: int,
    mean: tuple = IMAGENET_NORMALIZATION_MEAN,
    std: tuple = IMAGENET_NORMALIZATION_STD,
    augmentation: str = "mild",
) -> transforms.Compose:
    """Mild acquisition variation without horizontal flips or aggressive crops."""
    if augmentation == "acquisition":
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.90, 1.0), ratio=(0.95, 1.05)),
                transforms.RandomAffine(degrees=7, translate=(0.05, 0.05), scale=(0.95, 1.05)),
                transforms.ColorJitter(brightness=0.3, contrast=0.3),
                transforms.RandomAutocontrast(p=0.3),
                transforms.RandomAdjustSharpness(sharpness_factor=0.5, p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
    if augmentation != "mild":
        raise ValueError(f"Unknown augmentation: {augmentation}")
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomAffine(degrees=7, translate=(0.04, 0.04), scale=(0.95, 1.05)),
            transforms.ColorJitter(brightness=0.12, contrast=0.12),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def validate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple:
    """Collect validation logits with deterministic evaluation transforms."""
    model.eval()
    logits_list, labels_list = [], []
    with torch.inference_mode():
        for images, labels in loader:
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(images.to(device))
            logits_list.append(logits.float().cpu())
            labels_list.append(labels)
    logits, labels = torch.cat(logits_list), torch.cat(labels_list)
    loss = float(nn.functional.cross_entropy(logits, labels))
    return labels.numpy(), logits.softmax(1)[:, 1].numpy(), loss, logits.numpy()


def train(args: argparse.Namespace) -> dict:
    """Run a new experiment; persist its best model, protocol, history and scores."""
    if args.output.exists():
        raise FileExistsError(f"Experiment already exists: {args.output}; choose a new name")
    audit = verify_manifests(args.manifests)
    nih_task = getattr(args, "task", "legacy_pneumonia") == "nih_report_pneumonia"
    label_mapping = NIH_LABEL_TO_ID if nih_task else LABEL_TO_ID
    if nih_task:
        if audit.get("class_to_idx") != NIH_LABEL_TO_ID:
            raise ValueError("NIH task requires the explicitly labeled NIH split audit")
        if audit.get("fresh_test_exposed_patient_overlap") != 0:
            raise ValueError("NIH audit does not establish a fresh patient-disjoint test")
        if audit.get("historical_patient_roles_preserved") is not True:
            raise ValueError("NIH audit must preserve previous training/validation roles")
        if args.minimum_recall != 0.9:
            raise ValueError("The prospective NIH policy fixes minimum validation recall at 0.90")
        if args.model == "xrv_densenet121" or args.feature_checkpoint is not None:
            raise ValueError("Prospective NIH candidates require ImageNet-only initialization")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(4)
    set_global_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean = (0.5, 0.5, 0.5) if args.model == "xrv_densenet121" else IMAGENET_NORMALIZATION_MEAN
    std = (1 / 2048,) * 3 if args.model == "xrv_densenet121" else IMAGENET_NORMALIZATION_STD
    eval_transform = transforms.Compose(
        [
            transforms.Resize((args.size, args.size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    dataset_class = NIHReportDataset if nih_task else ChestXRayDataset
    train_data = dataset_class(
        args.manifests / "train_manifest.csv",
        train_transform(args.size, mean, std, args.augmentation),
    )
    val_data = dataset_class(args.manifests / "val_manifest.csv", eval_transform)
    if args.cache_images or args.preprocessing != "resize":
        print("Preparing lossless image cache", flush=True)
        cache_prepared_images(train_data, args.size, args.preprocessing)
        cache_prepared_images(val_data, args.size, args.preprocessing)
    generator = torch.Generator().manual_seed(args.seed)
    sampler = None
    if getattr(args, "balanced_sampling", False):
        if not nih_task or args.weighted_loss:
            raise ValueError("Balanced NIH sampling must not be combined with weighted loss")
        counts = np.bincount([s.label_id for s in train_data.samples], minlength=2)
        if counts.min() == 0:
            raise ValueError("Both training classes are required for balanced sampling")
        sample_weights = [1 / counts[s.label_id] for s in train_data.samples]
        sampler = WeightedRandomSampler(
            sample_weights, args.epoch_samples, replacement=True, generator=generator
        )
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=0,
        pin_memory=device.type == "cuda",
        generator=generator,
    )
    val_loader = DataLoader(
        val_data, batch_size=args.batch_size, num_workers=0, pin_memory=device.type == "cuda"
    )
    model = build_model(args.model, 2, pretrained=True, freeze_backbone=True).to(device)
    if args.feature_checkpoint is not None:
        if args.model != "resnet18":
            raise ValueError("RSNA feature transfer currently supports ResNet18 only")
        source = torch.load(args.feature_checkpoint, map_location="cpu", weights_only=False)
        if source["model_name"] != "resnet18":
            raise ValueError("Feature checkpoint architecture mismatch")
        if any(key.startswith("fc.") for key in source["feature_state_dict"]):
            raise ValueError("Opacity classifier must not be transferred to pneumonia")
        transferred = model.load_state_dict(source["feature_state_dict"], strict=False)
        if set(transferred.missing_keys) != {"fc.weight", "fc.bias"} or transferred.unexpected_keys:
            raise ValueError("Feature checkpoint does not match the complete backbone")
        model.pretraining_metadata = {
            "provider": "Audited RSNA development pretraining",
            "sha256": hashlib.sha256(args.feature_checkpoint.read_bytes()).hexdigest(),
            "protocol": source["pretraining_protocol"],
            "classifier_reset": True,
        }
    classifier = get_classifier_module(model)
    head_ids = {id(p) for p in classifier.parameters()}
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [p for p in model.parameters() if id(p) not in head_ids],
                "lr": args.lr * 0.2,
            },
            {"params": list(classifier.parameters()), "lr": args.lr},
        ],
        weight_decay=1e-4,
    )
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    weights = None
    if args.weighted_loss:
        counts = np.bincount([s.label_id for s in train_data.samples], minlength=2)
        weights = torch.tensor(
            np.sqrt(counts.sum() / (2 * counts)), dtype=torch.float32, device=device
        )
    criterion = nn.CrossEntropyLoss(weight=weights)
    args.output.mkdir(parents=True)
    protocol = {
        key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
    }
    protocol.update(
        {
            "data_audit": audit,
            "pretraining_metadata": getattr(model, "pretraining_metadata", None),
            "torch": torch.__version__,
            "python": platform.python_version(),
            "device": str(device),
            "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
            "checkpoint_selection": (
                "highest validation average precision, then lowest validation loss"
                if nih_task
                else "lowest unweighted validation cross entropy"
            ),
            "threshold_selection": (
                "validation F1 at recall >= minimum_recall"
                if nih_task
                else "validation specificity at recall >= minimum_recall"
            ),
            "label_mapping": label_mapping,
            "test_access": "none",
            "source_sha256": {
                name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                for name in (
                    "src/training/train_clean.py",
                    "src/training/models.py",
                    "src/data/image_preparation.py",
                    "src/data/dataset.py",
                    "src/data/nih_dataset.py",
                    "src/evaluation/decision.py",
                )
            },
            "pretraining": (
                "TorchXRayVision densenet121-res224-all"
                if args.model == "xrv_densenet121"
                else "torchvision ImageNet DEFAULT weights"
            ),
        }
    )
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    history, best_loss, stale = [], float("inf"), 0
    best_average_precision = -1.0
    start = time.perf_counter()
    for epoch in range(args.epochs):
        if epoch == args.warmup_epochs:
            for parameter in model.parameters():
                parameter.requires_grad = True
        model.train()
        if epoch < args.warmup_epochs:
            # Frozen ImageNet BatchNorm buffers must not drift during head warmup.
            model.eval()
            classifier.train()
        fraction = max(0, epoch - args.warmup_epochs) / max(args.epochs - args.warmup_epochs, 1)
        multiplier = 0.05 + 0.95 * (1 + math.cos(math.pi * fraction)) / 2
        optimizer.param_groups[0]["lr"] = args.lr * 0.2 * multiplier
        optimizer.param_groups[1]["lr"] = args.lr * multiplier
        loss_sum, samples = 0.0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach()) * len(labels)
            samples += len(labels)
        labels, probabilities, val_loss, val_logits = validate(model, val_loader, device)
        threshold_selector = select_f1_threshold if nih_task else select_threshold
        threshold = threshold_selector(labels, probabilities, args.minimum_recall)
        metrics = binary_metrics(labels, probabilities, threshold)
        entry = {
            "epoch": epoch + 1,
            "train_loss": loss_sum / samples,
            "val_loss": val_loss,
            "elapsed_seconds": time.perf_counter() - start,
            "validation": metrics,
            "validation_at_0_5": binary_metrics(labels, probabilities, 0.5),
        }
        history.append(entry)
        (args.output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(json.dumps(entry), flush=True)
        improved = (
            (metrics["average_precision"], -val_loss) > (best_average_precision, -best_loss)
            if nih_task
            else val_loss < best_loss
        )
        if improved:
            best_loss, stale = val_loss, 0
            best_average_precision = metrics["average_precision"]
            checkpoint = {
                "model_name": args.model,
                "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "label_mapping": label_mapping,
                "image_size": args.size,
                "normalization": {
                    "mean": mean,
                    "std": std,
                },
                "preprocessing": args.preprocessing,
                "decision_threshold": threshold,
                "model_version": args.output.name,
                "best_epoch": epoch + 1,
                "validation_metrics": metrics,
                "validation_loss": val_loss,
                "training_protocol": protocol,
            }
            temporary = args.output / "checkpoint.tmp"
            torch.save(checkpoint, temporary)
            temporary.replace(args.output / "best_model.pt")
            np.savez(
                args.output / "validation_predictions.npz",
                labels=labels,
                probabilities=probabilities,
                logits=val_logits,
            )
        else:
            stale += 1
        if epoch >= args.warmup_epochs + 5 and stale >= args.patience:
            print("Early stopping on the registered validation criterion", flush=True)
            break
    result = {
        "status": "complete",
        "best_validation_loss": best_loss,
        "selected_validation_average_precision": best_average_precision,
        "epochs": len(history),
        "elapsed_seconds": time.perf_counter() - start,
    }
    (args.output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    """CLI for a validation-only experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifests", type=Path, default=Path("data/processed/clean_v1"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model",
        choices=["resnet18", "efficientnet_b0", "efficientnet_v2_s", "xrv_densenet121"],
        default="resnet18",
    )
    parser.add_argument("--augmentation", choices=["mild", "acquisition"], default="mild")
    parser.add_argument("--feature-checkpoint", type=Path)
    parser.add_argument("--preprocessing", choices=["resize", "clahe"], default="resize")
    parser.add_argument("--cache-images", action="store_true")
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=22)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--minimum-recall", type=float, default=0.99)
    parser.add_argument("--weighted-loss", action="store_true")
    parser.add_argument(
        "--task", choices=["legacy_pneumonia", "nih_report_pneumonia"], default="legacy_pneumonia"
    )
    parser.add_argument("--balanced-sampling", action="store_true")
    parser.add_argument("--epoch-samples", type=int, default=16000)
    print(json.dumps(train(parser.parse_args())), flush=True)


if __name__ == "__main__":
    main()
