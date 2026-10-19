"""Budget-timed DINOv3 development training; no test manifest or test-image access."""

from __future__ import annotations

import hashlib
import json
import platform
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from src.cloud.dinov3_job import PILOT_TIMING_SCHEMA, digest, recipe, write_json
from src.core.reproducibility import set_global_seed
from src.data.dataset import LABEL_TO_ID, ChestXRayDataset
from src.data.transforms import IMAGENET_NORMALIZATION_MEAN, IMAGENET_NORMALIZATION_STD
from src.evaluation.decision import binary_metrics, select_threshold
from src.training.dinov3_model import DINOv3Classifier
from src.training.train_clean import train_transform, verify_manifests


def require_finite(value: torch.Tensor, description: str) -> None:
    """Reject numerical failure before exporting a model or updating parameters."""
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"Nonfinite {description}")


def loaders(manifests: Path, config: dict) -> tuple[DataLoader, DataLoader]:
    """Reuse the project's exact source-hash verification and mild augmentation."""
    verify_manifests(manifests)
    validation = transforms.Compose(
        [
            transforms.Resize((config["image_size"], config["image_size"])),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_NORMALIZATION_MEAN, IMAGENET_NORMALIZATION_STD),
        ]
    )
    datasets = (
        ChestXRayDataset(manifests / "train_manifest.csv", train_transform(config["image_size"])),
        ChestXRayDataset(manifests / "val_manifest.csv", validation),
    )
    if any({sample.label_id for sample in data.samples} != {0, 1} for data in datasets):
        raise ValueError("Both development splits must contain both classes")
    worker_options = (
        {"multiprocessing_context": "spawn", "timeout": 120} if config["num_workers"] else {}
    )
    return tuple(
        DataLoader(
            data,
            batch_size=config["batch_size"],
            shuffle=index == 0,
            num_workers=config["num_workers"],
            pin_memory=torch.cuda.is_available(),
            **worker_options,
        )
        for index, data in enumerate(datasets)
    )


def validation_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    deadline: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Collect full validation outputs, never select on a timed-out partial cohort."""
    model.eval()
    outputs, targets = [], []
    with torch.inference_mode():
        for images, labels in loader:
            if time.monotonic() >= deadline:
                raise TimeoutError("Training deadline reached during validation")
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(images.to(device))
            require_finite(logits, "validation logits")
            outputs.append(logits.float().cpu())
            targets.append(labels)
    logits, labels = torch.cat(outputs), torch.cat(targets)
    loss = nn.functional.cross_entropy(logits, labels)
    require_finite(loss, "validation loss")
    return labels.numpy(), logits.softmax(1)[:, 1].numpy(), float(loss)


def save_checkpoint(path: Path, value: dict) -> None:
    """Atomically persist a full checkpoint, including all resume state when present."""
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def run_training(
    config: dict,
    manifests: Path,
    weights: Path,
    output: Path,
    max_seconds: float,
    commit: Callable[[], None] | None = None,
    model_factory: Callable[[Path], DINOv3Classifier] = DINOv3Classifier.from_pretrained,
) -> dict:
    """Warm up the head then train the final blocks; resume only an identical recipe."""
    if max_seconds < 360:
        raise ValueError("Allow at least six minutes including checkpoint shutdown reserve")
    started = time.monotonic()
    deadline = started + max_seconds - 180
    output.mkdir(parents=True, exist_ok=True)
    set_global_seed(config["seed"])
    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = loaders(manifests, config)
    protocol = {
        "recipe": recipe(config),
        "test_access": "none",
        "manifest_sha256": {
            split: digest((manifests / f"{split}_manifest.csv").read_bytes())
            for split in ("train", "val")
        },
        "checkpoint_selection": "lowest full-validation cross entropy",
        "threshold_selection": "validation specificity at recall >= minimum_recall",
        "preprocessing": "resize; RGB; ImageNet normalization; no CLAHE",
        "feature_pooling": "CLS concatenated with mean patch tokens, excluding registers",
        "torch": torch.__version__,
        "python": platform.python_version(),
        "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
        "source_sha256": {
            name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
            for name in ("src/training/train_dinov3.py", "src/training/dinov3_model.py")
        },
        "limitations": "Previously observed legacy test; group IDs are filename proxies",
    }
    protocol_path = output / "protocol.json"
    if protocol_path.exists():
        previous = json.loads(protocol_path.read_text(encoding="utf-8"))
        for key in ("recipe", "manifest_sha256", "source_sha256"):
            if previous[key] != protocol[key]:
                raise ValueError(f"Resume protocol changed: {key}")
    else:
        write_json(protocol_path, protocol)
    model = model_factory(weights).to(device)
    tail_parameters = (
        [p for block in model.backbone.layer[-config["tail_blocks"] :] for p in block.parameters()]
        if config["tail_blocks"]
        else []
    )
    if config["tail_blocks"]:
        tail_parameters += list(model.backbone.norm.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": model.classifier.parameters(), "lr": config["head_lr"]},
            {"params": tail_parameters, "lr": config["backbone_lr"]},
        ],
        weight_decay=config["weight_decay"],
    )
    history, next_epoch, best_loss, stale = [], 0, float("inf"), 0
    checkpoint_path = output / "last.pt"
    if checkpoint_path.exists():
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if saved["protocol"]["recipe"] != protocol["recipe"]:
            raise ValueError("Resume checkpoint recipe mismatch")
        model.load_state_dict(saved["model_state_dict"], strict=True)
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        history, next_epoch = saved["history"], saved["next_epoch"]
        best_loss, stale = saved["best_loss"], saved["stale"]
        if any(not torch.isfinite(p).all() for p in model.parameters()):
            raise FloatingPointError("Nonfinite resume parameters")
    status = "complete"
    for epoch in range(next_epoch, config["epochs"]):
        # An interrupted epoch is replayed with the same seed from the last complete checkpoint.
        set_global_seed(config["seed"] + epoch)
        if stale >= config["patience"]:
            status = "early_stopped"
            break
        recent_seconds = history[-1]["epoch_seconds"] if history else 0
        if time.monotonic() + recent_seconds >= deadline:
            status = "time_limit"
            break
        epoch_start = time.monotonic()
        model.configure_training(0 if epoch < config["warmup_epochs"] else config["tail_blocks"])
        epoch_loss, count = 0.0, 0
        try:
            for images, labels in train_loader:
                if time.monotonic() >= deadline:
                    raise TimeoutError("Training deadline reached")
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
                ):
                    logits = model(images)
                    loss = nn.functional.cross_entropy(logits, labels)
                require_finite(logits, "training logits")
                require_finite(loss, "training loss")
                loss.backward()
                nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 5.0, error_if_nonfinite=True
                )
                optimizer.step()
                epoch_loss += float(loss.detach()) * len(labels)
                count += len(labels)
            labels, scores, val_loss = validation_predictions(model, val_loader, device, deadline)
        except TimeoutError:
            status = "time_limit"
            break
        threshold = select_threshold(labels, scores, config["minimum_recall"])
        metrics = binary_metrics(labels, scores, threshold)
        entry = {
            "epoch": epoch + 1,
            "train_loss": epoch_loss / count,
            "validation_loss": val_loss,
            "validation": metrics,
        }
        if val_loss < best_loss:
            best_loss, stale = val_loss, 0
            for parameter in model.parameters():
                require_finite(parameter, "checkpoint parameters")
            save_checkpoint(
                output / "best_model.pt",
                {
                    "format": "dinov3_binary_v1",
                    "backbone_config": model.backbone.config.to_dict(),
                    "model_state_dict": {
                        k: v.detach().cpu() for k, v in model.state_dict().items()
                    },
                    "label_mapping": LABEL_TO_ID,
                    "image_size": config["image_size"],
                    "normalization": {
                        "mean": IMAGENET_NORMALIZATION_MEAN,
                        "std": IMAGENET_NORMALIZATION_STD,
                    },
                    "preprocessing": "resize",
                    "decision_threshold": threshold,
                    "best_epoch": epoch + 1,
                    "validation_metrics": metrics,
                    "protocol": protocol,
                },
            )
            np.savez(output / "validation_predictions.npz", labels=labels, probabilities=scores)
        else:
            stale += 1
        history.append(entry)
        entry["epoch_seconds"] = time.monotonic() - epoch_start + 120
        save_checkpoint(
            checkpoint_path,
            {
                "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "optimizer_state_dict": optimizer.state_dict(),
                "next_epoch": epoch + 1,
                "best_loss": best_loss,
                "stale": stale,
                "history": history,
                "protocol": protocol,
            },
        )
        if commit is not None:
            commit()
        entry["epoch_seconds"] = time.monotonic() - epoch_start
        # Store timing after IO too, so the next epoch admission accounts for checkpoint cost.
        write_json(output / "history.json", history)
        print(json.dumps(entry), flush=True)
    result = {
        "status": status,
        "completed_epochs": len(history),
        "best_validation_loss": best_loss if history else None,
        "elapsed_seconds": time.monotonic() - started,
        "test_access": "none",
        "checkpoint": str(output / "best_model.pt") if history else None,
    }
    write_json(output / "result.json", result)
    if commit is not None:
        commit()
    return result


def run_pilot(
    config: dict,
    manifests: Path,
    weights: Path,
    output: Path,
    progress: Callable[[dict], None] | None = None,
) -> dict:
    """Measure tail-training throughput without selecting a model on partial validation."""
    pilot_started = time.monotonic()

    def report(stage: str, **details: object) -> None:
        event = {"stage": stage, "elapsed_seconds": time.monotonic() - pilot_started, **details}
        if progress is not None:
            progress(event)
        else:
            print(json.dumps(event), flush=True)

    report("initialize_cuda_and_seed")
    set_global_seed(config["seed"])
    torch.set_num_threads(2)
    report("verify_images_and_build_loaders")
    train_loader, val_loader = loaders(manifests, config)
    device = torch.device("cuda")
    report("load_pretrained_model")
    model = DINOv3Classifier.from_pretrained(weights).to(device)
    model.configure_training(config["tail_blocks"])
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=config["backbone_lr"]
    )
    torch.cuda.reset_peak_memory_stats()
    durations = []
    report("start_training_workers", gpu=torch.cuda.get_device_name())
    iterator = iter(train_loader)
    setup_seconds = time.monotonic() - pilot_started
    warmup_batches = min(max(2, config["num_workers"] + 1), len(train_loader) - 1)
    loader_warmup_seconds = 0.0
    for index in range(min(warmup_batches + 6, len(train_loader))):
        if index < warmup_batches:
            report("load_training_batch", batch=index + 1)
        elif index == warmup_batches:
            report("measure_training_batches")
        torch.cuda.synchronize()
        started = time.monotonic()
        images, labels = next(iterator)
        data_seconds = time.monotonic() - started
        if index < warmup_batches:
            report("compute_training_batch", batch=index + 1)
        # Progress commits are useful diagnostics, but training does not perform
        # them per batch. Exclude their IO cost from the throughput forecast.
        compute_started = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(images.to(device))
            loss = nn.functional.cross_entropy(logits, labels.to(device))
        require_finite(loss, "pilot loss")
        loss.backward()
        nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 5.0, error_if_nonfinite=True
        )
        optimizer.step()
        torch.cuda.synchronize()
        duration = data_seconds + time.monotonic() - compute_started
        if index >= warmup_batches:
            durations.append(duration)
        else:
            loader_warmup_seconds += duration
        if index < warmup_batches:
            report("training_batch_complete", batch=index + 1, batch_seconds=duration)
    if not durations:
        raise ValueError("Pilot requires at least two training batches")
    report("training_measurement_complete", batch_seconds=durations)
    model.eval()
    report("validation_batch")
    warmup_started = time.monotonic()
    validation_iterator = iter(val_loader)
    validation_warmup = min(max(2, config["num_workers"]), len(val_loader) - 1)
    validation_durations = []
    for index in range(min(validation_warmup + 6, len(val_loader))):
        started = time.monotonic()
        images, _ = next(validation_iterator)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            require_finite(model(images.to(device)), "pilot validation logits")
        torch.cuda.synchronize()
        if index < validation_warmup:
            if index + 1 == validation_warmup:
                loader_warmup_seconds += time.monotonic() - warmup_started
        else:
            validation_durations.append(time.monotonic() - started)
    if not validation_durations:
        raise ValueError("Pilot requires at least two validation batches")
    val_seconds = max(validation_durations)
    batch_seconds = max(durations)
    epoch_estimate = (
        batch_seconds * len(train_loader)
        + val_seconds * len(val_loader)
        + loader_warmup_seconds
        + 120
    )
    result = {
        "timing_schema": PILOT_TIMING_SCHEMA,
        "setup_seconds": setup_seconds,
        "loader_warmup_seconds": loader_warmup_seconds,
        "train_batch_seconds": batch_seconds,
        "validation_batch_seconds": val_seconds,
        "conservative_epoch_seconds": epoch_estimate,
        "estimated_training_seconds": setup_seconds + epoch_estimate * config["epochs"],
        "peak_gpu_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "test_access": "none",
        "partial_validation_metrics": "not computed",
    }
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "pilot.json", result)
    report("pilot_complete")
    return result
