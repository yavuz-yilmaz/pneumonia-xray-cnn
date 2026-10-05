"""Fit validation-selected linear heads on external chest-X-ray representations.

Legacy runs use mixed TorchXRayVision features. NIH runs require the separately
verified MIMIC-only feature artifact and retain NIH report-label semantics.
This training command opens only training and validation manifests and images.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from torch.utils.data import DataLoader
from torchvision import transforms

from src.core.reproducibility import set_global_seed
from src.data.dataset import LABEL_TO_ID, ChestXRayDataset
from src.data.image_preparation import cache_prepared_images
from src.data.nih_dataset import NIH_LABEL_TO_ID, NIHReportDataset
from src.evaluation.decision import binary_metrics, select_f1_threshold, select_threshold
from src.training.models import XRayDenseNet121
from src.training.train_clean import verify_manifests

XRAY_MEAN = (0.5, 0.5, 0.5)
XRAY_STD = (1 / 2048, 1 / 2048, 1 / 2048)
EXTERNAL_WEIGHTS_SHA256 = "56524913dd16a906422e8d8b66a7a5c46be1d82eb7ac012d8103776f1aa68899"
EXTERNAL_WEIGHTS_NAME = (
    "nih-pc-chex-mimic_ch-google-openi-kaggle-densenet121-"
    "d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt"
)
MIMIC_SOURCE_SHA256 = "23b13a04459684ffa41247d068207a7657b4e1b4dec2b02431f5026bd75b1189"
MIMIC_FEATURE_SHA256 = "7c5df8cf35f00b01d2b90e77f4301027c9d1555395a0c15512aaf2947973e7c0"
MIMIC_WEIGHTS_NAME = "mimic_ch-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt"


def load_pretrained_features(pretraining: str, checkpoint: Path | None) -> XRayDenseNet121:
    """Pin the specific pretraining source before constructing a frozen backbone."""
    if pretraining == "all":
        if checkpoint is not None:
            raise ValueError("Mixed pretraining does not accept a replacement feature checkpoint")
        external = Path.home() / ".torchxrayvision" / "models_data" / EXTERNAL_WEIGHTS_NAME
        if hashlib.sha256(external.read_bytes()).hexdigest() != EXTERNAL_WEIGHTS_SHA256:
            raise ValueError("External weights differ from the reviewed version")
        return XRayDenseNet121(pretrained=True)
    if pretraining != "mimic_ch" or checkpoint is None:
        raise ValueError("MIMIC-specific pretraining requires its verified feature checkpoint")
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != MIMIC_FEATURE_SHA256:
        raise ValueError("MIMIC feature checkpoint differs from the reviewed tensor artifact")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    metadata = saved["pretraining_metadata"]
    if (
        metadata.get("weights") != "densenet121-res224-mimic_ch"
        or metadata.get("source_sha256") != MIMIC_SOURCE_SHA256
    ):
        raise ValueError("Feature pretraining provenance is not the approved MIMIC source")
    model = XRayDenseNet121(pretrained=False)
    model.features.load_state_dict(saved["feature_state_dict"], strict=True)
    model.pretraining_metadata = dict(metadata, feature_checkpoint_sha256=MIMIC_FEATURE_SHA256)
    return model


def extract(
    model: XRayDenseNet121,
    manifest: Path,
    device: torch.device,
    dataset_class: type[ChestXRayDataset] = ChestXRayDataset,
    cache_images: bool = False,
) -> tuple:
    """Extract frozen full-precision features once per audited development image."""
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(XRAY_MEAN, XRAY_STD),
        ]
    )
    dataset = dataset_class(manifest, transform)
    if cache_images:
        cache_prepared_images(dataset, 224, "resize")
    loader = DataLoader(dataset, batch_size=24, num_workers=0, shuffle=False)
    all_features, all_labels = [], []
    model.eval()
    with torch.inference_mode():
        for index, (images, labels) in enumerate(loader):
            all_features.append(model.extract_features(images.to(device)).cpu().numpy())
            all_labels.append(labels.numpy())
            if index % 25 == 0:
                print(
                    f"{manifest.name}: {min((index + 1) * 24, len(dataset))}/{len(dataset)}",
                    flush=True,
                )
    return np.concatenate(all_features), np.concatenate(all_labels)


def train(args: argparse.Namespace) -> dict:
    """Select regularization using validation while preserving all candidate records."""
    if args.output.exists():
        raise FileExistsError(f"Experiment directory already exists: {args.output}")
    task = getattr(args, "task", "legacy_pneumonia")
    pretraining = getattr(args, "pretraining", "all")
    nih_task = task == "nih_report_pneumonia"
    if nih_task and pretraining != "mimic_ch":
        raise ValueError("NIH evaluation cannot use mixed NIH/OpenI-pretrained features")
    audit = verify_manifests(args.manifests)
    if nih_task and (
        audit.get("class_to_idx") != NIH_LABEL_TO_ID
        or audit.get("fresh_test_exposed_patient_overlap") != 0
        or audit.get("historical_patient_roles_preserved") is not True
    ):
        raise ValueError("NIH probe requires its audited patient-disjoint report-label cohort")
    torch.set_num_threads(4)
    set_global_seed(42)
    requested_device = getattr(args, "device", "auto")
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if requested_device == "auto"
        else torch.device(requested_device)
    )
    model = (
        load_pretrained_features(pretraining, getattr(args, "feature_checkpoint", None))
        .to(device)
        .eval()
    )
    for parameter in model.features.parameters():
        parameter.requires_grad = False
    args.output.mkdir(parents=True)
    start = time.perf_counter()
    dataset_class = NIHReportDataset if nih_task else ChestXRayDataset
    cache_images = getattr(args, "cache_images", False)
    train_features, train_labels = extract(
        model, args.manifests / "train_manifest.csv", device, dataset_class, cache_images
    )
    val_features, val_labels = extract(
        model, args.manifests / "val_manifest.csv", device, dataset_class, cache_images
    )
    np.savez(
        args.output / "features.npz",
        train=train_features,
        train_labels=train_labels,
        val=val_features,
        val_labels=val_labels,
    )
    scaler = StandardScaler().fit(train_features)
    standardized = scaler.transform(train_features)
    candidates = []
    regularizations = (0.001, 0.01, 0.1, 1.0) if nih_task else (0.01, 0.1, 1.0, 10.0)
    for regularization in regularizations:
        with threadpool_limits(limits=4):
            classifier = LogisticRegression(C=regularization, solver="lbfgs", max_iter=2000)
            classifier.fit(standardized, train_labels)
        if int(classifier.n_iter_.max()) >= 2000:
            raise RuntimeError("Logistic regression did not converge")
        coefficient = classifier.coef_[0] / scaler.scale_
        intercept = classifier.intercept_[0] - float(np.dot(coefficient, scaler.mean_))
        with torch.no_grad():
            model.classifier.weight.zero_()
            model.classifier.bias.zero_()
            model.classifier.weight[1].copy_(torch.from_numpy(coefficient).float())
            model.classifier.bias[1] = intercept
            logits = model.classifier(torch.from_numpy(val_features).to(device))
            scores = logits.softmax(1)[:, 1].cpu().numpy()
        minimum_recall = 0.9 if nih_task else 0.99
        selector = select_f1_threshold if nih_task else select_threshold
        threshold = selector(val_labels, scores, minimum_recall=minimum_recall)
        metrics = binary_metrics(val_labels, scores, threshold)
        directory = args.output / f"c{str(regularization).replace('.', 'p')}"
        directory.mkdir()
        protocol = {
            "model": "xrv_densenet121",
            "backbone_training": "frozen",
            "C": regularization,
            "minimum_recall": minimum_recall,
            "task": task,
            "pretraining": pretraining,
            "data_audit": audit,
            "external_weights_sha256": (
                MIMIC_SOURCE_SHA256 if pretraining == "mimic_ch" else EXTERNAL_WEIGHTS_SHA256
            ),
            "external_weights_url": "https://github.com/mlmed/torchxrayvision/releases/download/v1/"
            + (MIMIC_WEIGHTS_NAME if pretraining == "mimic_ch" else EXTERNAL_WEIGHTS_NAME),
            "pretraining_metadata": getattr(model, "pretraining_metadata", None),
            "test_access": "none",
            "feature_scaler_fit": "train only",
            "prior_research": (
                "NIH v1 validation observed; this command never reads NIH/OpenI test predictions"
                if nih_task
                else "clean_v1 legacy test failed; v2 is a separately recorded experiment"
            ),
            "checkpoint_selection": (
                "highest validation average precision; lowest C breaks ties"
                if nih_task
                else "legacy selection remains separate"
            ),
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "training_method": (
                "standardized frozen features; L2 logistic head; scaler folded into head"
            ),
        }
        checkpoint = {
            "model_name": "xrv_densenet121",
            "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "image_size": 224,
            "normalization": {"mean": XRAY_MEAN, "std": XRAY_STD},
            "label_mapping": NIH_LABEL_TO_ID if nih_task else LABEL_TO_ID,
            "decision_threshold": threshold,
            "model_version": (
                f"nih_mimic_frozen_{directory.name}"
                if nih_task
                else f"clean_v2_xrv_frozen_{directory.name}"
            ),
            "validation_metrics": metrics,
            "training_protocol": protocol,
        }
        torch.save(checkpoint, directory / "best_model.pt")
        np.savez(directory / "validation_predictions.npz", labels=val_labels, probabilities=scores)
        (directory / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
        (directory / "result.json").write_text(
            json.dumps(
                {
                    "status": "complete",
                    "iterations": int(classifier.n_iter_.max()),
                    "validation": metrics,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        candidates.append(
            {"directory": directory.as_posix(), "validation": metrics, "C": regularization}
        )
        print(json.dumps(candidates[-1]), flush=True)
    result = {
        "status": "complete",
        "candidates": candidates,
        "elapsed_seconds": time.perf_counter() - start,
    }
    if nih_task:
        result["selected_candidate"] = max(
            candidates,
            key=lambda candidate: (candidate["validation"]["average_precision"], -candidate["C"]),
        )
    (args.output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    """Run the prespecified four-value regularization comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifests", type=Path, default=Path("data/processed/clean_v1"))
    parser.add_argument("--output", type=Path, default=Path("models/clean_v2/xrv_frozen"))
    parser.add_argument(
        "--task", choices=["legacy_pneumonia", "nih_report_pneumonia"], default="legacy_pneumonia"
    )
    parser.add_argument("--pretraining", choices=["all", "mimic_ch"], default="all")
    parser.add_argument("--feature-checkpoint", type=Path)
    parser.add_argument("--cache-images", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    print(json.dumps(train(args), indent=2), flush=True)


if __name__ == "__main__":
    main()
