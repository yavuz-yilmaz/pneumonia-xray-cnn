"""Adapt MIMIC-only DenseNet's final block on audited NIH development data."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.core.reproducibility import set_global_seed
from src.data.download_nih import file_sha256
from src.data.image_preparation import CheckpointImageTransform, cache_prepared_images
from src.data.nih_dataset import NIH_LABEL_TO_ID, NIHReportDataset
from src.evaluation.decision import binary_metrics, select_f1_threshold
from src.training.calibrate_joint_metrics import select_joint_threshold
from src.training.models import XRayDenseNet121
from src.training.train_clean import train_transform, verify_manifests
from src.training.train_xray_probe import XRAY_MEAN, XRAY_STD, load_pretrained_features

HEAD_SHA256 = "cfa814ced6dcf49a80cafca39a946edd8d25714fcfbdcdee881ca30624a04e31"


def configure_tail(model: XRayDenseNet121) -> None:
    """Train only the final dense block and head; preserve all BatchNorm buffers."""
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith(("features.denseblock4.", "classifier."))
    model.train()
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def load_initial_model(feature_path: Path, head_path: Path) -> XRayDenseNet121:
    """Verify the fixed NIH learned head and require the original MIMIC backbone."""
    if file_sha256(head_path) != HEAD_SHA256:
        raise ValueError("Initial learned checkpoint differs from the registered artifact")
    model = load_pretrained_features("mimic_ch", feature_path)
    checkpoint = torch.load(head_path, map_location="cpu", weights_only=True)
    if checkpoint["label_mapping"] != NIH_LABEL_TO_ID:
        raise ValueError("Initial head has different target semantics")
    for name, tensor in model.features.state_dict().items():
        if not torch.equal(tensor, checkpoint["model_state_dict"][f"features.{name}"]):
            raise ValueError("Initial learned head does not have the unchanged MIMIC backbone")
    model.classifier.load_state_dict(
        {name: checkpoint["model_state_dict"][f"classifier.{name}"] for name in ("weight", "bias")}
    )
    return model


def validation_predictions(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate deterministic validation images in float32 without test access."""
    labels, scores = [], []
    model.eval()
    with torch.inference_mode():
        for images, targets in loader:
            scores.append(model(images.to(device)).softmax(1)[:, 1].cpu().numpy())
            labels.append(targets.numpy())
    return np.concatenate(labels), np.concatenate(scores)


def train(args: argparse.Namespace) -> dict:
    """Run one fixed development experiment, preserving failures and selection."""
    if args.output.exists():
        raise FileExistsError("Experiment already exists; inspect the recorded process/state")
    args.output.mkdir(parents=True)
    status = {"process_id": os.getpid(), "state": "preflight", "test_access": "none"}

    def save_status() -> None:
        status["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        temporary = args.output / "status.tmp"
        temporary.write_text(json.dumps(status, indent=2), encoding="utf-8")
        temporary.replace(args.output / "status.json")

    save_status()
    try:
        protocol = {
            "registered_at_utc": datetime.now(timezone.utc).isoformat(),
            "task": "nih_report_pneumonia",
            "pretraining": "mimic_ch",
            "label_mapping": NIH_LABEL_TO_ID,
            "epochs": args.epochs,
            "epoch_samples": args.epoch_samples,
            "batch_size": args.batch_size,
            "seed": 20261002,
            "feature_lr": 1e-5,
            "head_lr": 1e-4,
            "weight_decay": 1e-4,
            "arithmetic": "float32 training and validation; reject nonfinite logits/loss/gradients",
            "sampling": "balanced with replacement; unweighted CE",
            "training_scope": "denseblock4 and classifier; all BatchNorm buffers frozen",
            "input": "224px resize; XRV normalization; mild acquisition augmentation",
            "selection": "highest validation AP; earliest epoch breaks ties",
            "threshold": "after AP selection, joint validation floors from both fixed baselines",
            "prior_exposure": (
                "NIH test and OpenI already observed; this is exploratory development"
            ),
            "test_access": "none",
            "openi_access": "none",
            "auto_deployment": False,
            "initial_head_sha256": HEAD_SHA256,
            "inputs_sha256": {
                str(path): file_sha256(path)
                for path in (
                    Path(__file__),
                    Path("docs/nih_transfer_protocol.md"),
                    Path("src/training/train_xray_probe.py"),
                    Path("src/training/models.py"),
                    Path("src/training/train_clean.py"),
                    Path("src/data/image_preparation.py"),
                    Path("src/data/dataset.py"),
                    Path("src/data/nih_dataset.py"),
                    Path("src/evaluation/decision.py"),
                    Path("src/training/calibrate_joint_metrics.py"),
                    args.manifests / "train_manifest.csv",
                    args.manifests / "val_manifest.csv",
                    args.manifests / "split_audit.json",
                    args.feature_checkpoint,
                    args.head_checkpoint,
                    args.baseline_result,
                    args.baseline_result.with_name("original_validation_predictions.npz"),
                    args.baseline_result.with_name("current_validation_predictions.npz"),
                )
            },
        }
        (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
        audit = verify_manifests(args.manifests)
        if (
            audit.get("class_to_idx") != NIH_LABEL_TO_ID
            or audit.get("historical_patient_roles_preserved") is not True
            or audit.get("fresh_test_exposed_patient_overlap") != 0
        ):
            raise ValueError("NIH patient/duplicate audit does not establish the required split")
        baseline = json.loads(args.baseline_result.read_bytes())
        if (
            baseline["validation_manifest_sha256"]
            != protocol["inputs_sha256"][str(args.manifests / "val_manifest.csv")]
        ):
            raise ValueError("Saved baselines used a different validation manifest")
        baselines = baseline["baselines"]
        with (args.manifests / "val_manifest.csv").open(encoding="utf-8", newline="") as stream:
            expected_labels = np.asarray([int(row["label_id"]) for row in csv.DictReader(stream)])
        for name in ("original", "current"):
            path = args.baseline_result.with_name(f"{name}_validation_predictions.npz")
            with np.load(path, allow_pickle=False) as saved:
                if not np.array_equal(saved["labels"], expected_labels):
                    raise ValueError("Saved baseline labels differ from validation manifest")
                recomputed = binary_metrics(
                    saved["labels"], saved["probabilities"], baselines[name]["threshold"]
                )
            if any(
                abs(recomputed[key] - baselines[name][key]) > 1e-12
                for key in ("accuracy", "precision", "recall", "f1")
            ):
                raise ValueError("Saved baseline metric floors disagree with their predictions")
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.set_num_threads(4)
        set_global_seed(20261002)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_initial_model(args.feature_checkpoint, args.head_checkpoint).to(device)
        train_data = NIHReportDataset(
            args.manifests / "train_manifest.csv", train_transform(224, XRAY_MEAN, XRAY_STD)
        )
        val_data = NIHReportDataset(
            args.manifests / "val_manifest.csv",
            CheckpointImageTransform(224, XRAY_MEAN, XRAY_STD),
        )
        cache_prepared_images(train_data, 224, "resize")
        cache_prepared_images(val_data, 224, "resize")
        counts = np.bincount([s.label_id for s in train_data.samples], minlength=2)
        generator = torch.Generator().manual_seed(20261002)
        sampler = WeightedRandomSampler(
            [1 / counts[s.label_id] for s in train_data.samples],
            args.epoch_samples,
            replacement=True,
            generator=generator,
        )
        train_loader = DataLoader(
            train_data,
            sampler=sampler,
            batch_size=args.batch_size,
            num_workers=0,
            pin_memory=device.type == "cuda",
            generator=generator,
        )
        val_loader = DataLoader(val_data, batch_size=24, num_workers=0)
        configure_tail(model)
        optimizer = torch.optim.AdamW(
            [
                {"params": model.features.denseblock4.parameters(), "lr": 1e-5},
                {"params": model.classifier.parameters(), "lr": 1e-4},
            ],
            weight_decay=1e-4,
        )
        history, best_ap, start = [], -1.0, time.perf_counter()
        frozen = {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
            if not name.startswith(("features.denseblock4.", "classifier."))
            or name.endswith(("running_mean", "running_var", "num_batches_tracked"))
        }
        status["state"] = "training"
        save_status()
        for epoch in range(args.epochs):
            configure_tail(model)
            loss_sum, samples = 0.0, 0
            for images, labels in train_loader:
                optimizer.zero_grad(set_to_none=True)
                logits = model(images.to(device))
                if not torch.isfinite(logits).all():
                    raise RuntimeError("Nonfinite training logits; preserve this failed run")
                loss = nn.functional.cross_entropy(logits, labels.to(device))
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite training loss; preserve this failed run")
                loss.backward()
                nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    5.0,
                    error_if_nonfinite=True,
                )
                optimizer.step()
                loss_sum += float(loss.detach()) * len(labels)
                samples += len(labels)
            labels, scores = validation_predictions(model, val_loader, device)
            if not np.array_equal(labels, expected_labels):
                raise ValueError("Validation predictions differ from immutable manifest order")
            threshold = select_f1_threshold(labels, scores, minimum_recall=0.9)
            metrics = binary_metrics(labels, scores, threshold)
            for name, tensor in frozen.items():
                if not torch.equal(tensor, model.state_dict()[name].detach().cpu()):
                    raise ValueError(f"Frozen tensor changed: {name}")
            entry = {
                "epoch": epoch + 1,
                "train_loss": loss_sum / samples,
                "validation_at_recall90": metrics,
                "elapsed_seconds": time.perf_counter() - start,
            }
            history.append(entry)
            (args.output / "history.json").write_text(
                json.dumps(history, indent=2), encoding="utf-8"
            )
            if metrics["average_precision"] > best_ap:
                best_ap = metrics["average_precision"]
                checkpoint = {
                    "model_name": "xrv_densenet121",
                    "model_state_dict": {
                        k: v.detach().cpu() for k, v in model.state_dict().items()
                    },
                    "label_mapping": NIH_LABEL_TO_ID,
                    "image_size": 224,
                    "preprocessing": "resize",
                    "normalization": {"mean": XRAY_MEAN, "std": XRAY_STD},
                    "decision_threshold": threshold,
                    "validation_metrics": metrics,
                    "model_version": "nih_mimic_tail_exploratory",
                    "best_epoch": epoch + 1,
                    "training_protocol": protocol,
                    "pretraining_metadata": model.pretraining_metadata,
                }
                torch.save(checkpoint, args.output / "best_model.pt")
                np.savez(
                    args.output / "validation_predictions.npz", labels=labels, probabilities=scores
                )
            status["completed_epochs"] = epoch + 1
            save_status()
            print(json.dumps(entry), flush=True)
        with np.load(args.output / "validation_predictions.npz", allow_pickle=False) as saved:
            labels, scores = saved["labels"], saved["probabilities"]
        joint_threshold = select_joint_threshold(labels, scores, baselines)
        result = {
            "status": "complete",
            "selected_validation_ap": best_ap,
            "joint_threshold": joint_threshold,
            "joint_validation": None
            if joint_threshold is None
            else binary_metrics(labels, scores, joint_threshold),
            "test_access": "none",
            "external_improvement_established": False,
            "elapsed_seconds": time.perf_counter() - start,
            "checkpoint_sha256": file_sha256(args.output / "best_model.pt"),
        }
        (args.output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        status["state"] = "complete_development_only"
        save_status()
        return result
    except Exception as error:
        status.update(state="failed_preserved", error=f"{type(error).__name__}: {error}")
        save_status()
        raise


def main() -> None:
    """Launch a new immutable development run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifests", type=Path, default=Path("data/processed/nih_pneumonia_v1"))
    parser.add_argument("--output", type=Path, default=Path("models/nih_mimic_tail_v2"))
    parser.add_argument(
        "--feature-checkpoint",
        type=Path,
        default=Path("models/pretrained/xrv_mimic_ch/features_safe.pt"),
    )
    parser.add_argument(
        "--head-checkpoint",
        type=Path,
        default=Path("models/nih_mimic_v1/frozen/c0p001/best_model.pt"),
    )
    parser.add_argument(
        "--baseline-result", type=Path, default=Path("models/nih_mimic_v1/joint_policy/result.json")
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--epoch-samples", type=int, default=12000)
    parser.add_argument("--batch-size", type=int, default=16)
    print(json.dumps(train(parser.parse_args()), indent=2), flush=True)


if __name__ == "__main__":
    main()
