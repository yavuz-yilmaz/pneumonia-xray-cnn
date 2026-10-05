"""Fit a registered SSMU classifier using only train and validation images."""

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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from src.core.reproducibility import set_global_seed
from src.data.dataset import LABEL_TO_ID
from src.data.download_nih import file_sha256
from src.evaluation.decision import binary_metrics, select_f1_threshold
from src.evaluation.evaluate_clean import collect_predictions
from src.inference.predict import load_model
from src.training.calibrate_joint_metrics import select_joint_threshold
from src.training.train_clean import verify_manifests
from src.training.train_xray_probe import (
    XRAY_MEAN,
    XRAY_STD,
    extract,
    load_pretrained_features,
)

BASELINE_HASHES = {
    "original": "6dd50d60330b52664236c74a45696a04770723e91c5dbea3715e322709a995d7",
    "current": "da5c58d36865cc01d8df1dce5c358dac2a06b0b0b4981115bb7e35d806e2be9b",
}
REGULARIZATIONS = (0.001, 0.01, 0.1, 1.0)


def fit_head(
    features: np.ndarray, labels: np.ndarray, validation: np.ndarray, regularization: float
) -> tuple[dict, np.ndarray, dict]:
    """Fit scaling on training only and verify its folded float32 classifier."""
    if (
        not np.isfinite(features).all()
        or not np.isfinite(validation).all()
        or set(np.unique(labels)) != {0, 1}
    ):
        raise ValueError("Invalid development features/labels")
    scaler = StandardScaler().fit(features)
    with threadpool_limits(limits=4):
        classifier = LogisticRegression(C=regularization, solver="lbfgs", max_iter=2000)
        classifier.fit(scaler.transform(features), labels)
    if int(classifier.n_iter_.max()) >= 2000:
        raise RuntimeError("Logistic regression failed to converge")
    coefficient = classifier.coef_[0] / scaler.scale_
    intercept = classifier.intercept_[0] - float(np.dot(coefficient, scaler.mean_))
    weight = torch.zeros((2, features.shape[1]), dtype=torch.float32)
    bias = torch.zeros(2, dtype=torch.float32)
    weight[1] = torch.from_numpy(coefficient).float()
    bias[1] = intercept
    scores = torch.nn.functional.linear(torch.from_numpy(validation).float(), weight, bias)
    scores = scores.softmax(1)[:, 1].numpy()
    reference = classifier.predict_proba(scaler.transform(validation))[:, 1]
    if not np.isfinite(scores).all() or not np.allclose(scores, reference, rtol=1e-4, atol=5e-5):
        raise ValueError("Folded head differs from the fitted classifier")
    return (
        {"weight": weight, "bias": bias},
        scores,
        {
            "iterations": int(classifier.n_iter_.max()),
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "folded_prediction_max_absolute_error": float(np.max(np.abs(scores - reference))),
        },
    )


def train(args: argparse.Namespace) -> dict:
    """Preserve fixed candidates, baseline predictions and infeasible outcomes."""
    if args.output.exists():
        raise FileExistsError("Preserve prior experiment; inspect actual status")
    args.output.mkdir(parents=True)
    status = {"process_id": os.getpid(), "state": "preflight", "test_access": "none"}

    def save_status() -> None:
        status["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        temporary = args.output / "status.tmp"
        temporary.write_text(json.dumps(status, indent=2), encoding="utf-8")
        temporary.replace(args.output / "status.json")

    save_status()
    try:
        baselines_paths = {"original": args.original, "current": args.current}
        for name, path in baselines_paths.items():
            if file_sha256(path) != BASELINE_HASHES[name]:
                raise ValueError("Original/current baseline differs from pinned checkpoint")
        inputs = {
            str(p): file_sha256(p)
            for p in (
                Path(__file__),
                Path("docs/ssmu_development_protocol.md"),
                args.manifests / "split_audit.json",
                args.manifests / "train_manifest.csv",
                args.manifests / "val_manifest.csv",
                args.feature_checkpoint,
                args.original,
                args.current,
                Path("src/data/dataset.py"),
                Path("src/data/image_preparation.py"),
                Path("src/training/train_xray_probe.py"),
                Path("src/training/models.py"),
                Path("src/evaluation/evaluate_clean.py"),
                Path("src/inference/predict.py"),
                Path("src/training/train_clean.py"),
                Path("src/training/calibrate_joint_metrics.py"),
                Path("src/evaluation/decision.py"),
            )
        }
        protocol = {
            "registered_at_utc": datetime.now(timezone.utc).isoformat(),
            "task": "ssmu_source_pneumonia",
            "regularizations": REGULARIZATIONS,
            "seed": 20261002,
            "input_sha256": inputs,
            "selection": "highest validation AP; lowest C on ties",
            "threshold": "joint accuracy/precision/recall/F1 baseline floors after selection",
            "scaler_fit": "training only",
            "pretraining": "verified MIMIC-only features",
            "test_access": "none",
            "patient_identity_verified": False,
            "clinical_labels_verified": False,
            "auto_deployment": False,
        }
        (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
        audit = verify_manifests(args.manifests)
        if audit.get("task") != "ssmu_source_pneumonia" or audit.get("class_to_idx") != LABEL_TO_ID:
            raise ValueError("SSMU source/task audit required")
        with (args.manifests / "val_manifest.csv").open(newline="", encoding="utf-8") as stream:
            expected = np.asarray([int(r["label_id"]) for r in csv.DictReader(stream)])
        torch.set_num_threads(4)
        set_global_seed(20261002)
        baselines = {}
        status["state"] = "scoring_fixed_validation_baselines"
        save_status()
        for name, path in baselines_paths.items():
            loaded = load_model(path)
            threshold = loaded.decision_threshold
            del loaded
            labels, probabilities = collect_predictions(path, args.manifests / "val_manifest.csv")
            if not np.array_equal(labels, expected):
                raise ValueError("Baseline label order differs")
            baselines[name] = binary_metrics(labels, probabilities, threshold)
            np.savez(
                args.output / f"{name}_validation_predictions.npz",
                labels=labels,
                probabilities=probabilities,
            )
            print(json.dumps({"baseline": name, "validation": baselines[name]}), flush=True)
        (args.output / "baselines.json").write_text(
            json.dumps(baselines, indent=2), encoding="utf-8"
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_pretrained_features("mimic_ch", args.feature_checkpoint).to(device).eval()
        for parameter in model.parameters():
            parameter.requires_grad = False
        status["state"] = "extracting_development_features"
        save_status()
        start = time.perf_counter()
        training, training_labels = extract(
            model, args.manifests / "train_manifest.csv", device, cache_images=True
        )
        validation, validation_labels = extract(
            model, args.manifests / "val_manifest.csv", device, cache_images=True
        )
        if not np.array_equal(validation_labels, expected):
            raise ValueError("Feature validation labels differ")
        np.savez(
            args.output / "features.npz",
            train=training,
            train_labels=training_labels,
            val=validation,
            val_labels=validation_labels,
        )
        candidates = []
        status["state"] = "fitting_registered_heads"
        save_status()
        for regularization in REGULARIZATIONS:
            head, scores, details = fit_head(training, training_labels, validation, regularization)
            threshold = select_f1_threshold(validation_labels, scores, minimum_recall=0.9)
            metrics = binary_metrics(validation_labels, scores, threshold)
            folder = args.output / f"c{str(regularization).replace('.', 'p')}"
            folder.mkdir()
            model.classifier.load_state_dict(head)
            checkpoint = {
                "model_name": "xrv_densenet121",
                "label_mapping": LABEL_TO_ID,
                "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "image_size": 224,
                "preprocessing": "resize",
                "normalization": {"mean": XRAY_MEAN, "std": XRAY_STD},
                "decision_threshold": threshold,
                "validation_metrics": metrics,
                "model_version": f"ssmu_frozen_exploratory_c{regularization}",
                "training_protocol": protocol,
                "pretraining_metadata": model.pretraining_metadata,
            }
            torch.save(checkpoint, folder / "best_model.pt")
            np.savez(
                folder / "validation_predictions.npz",
                labels=validation_labels,
                probabilities=scores,
            )
            candidate = {
                "C": regularization,
                "directory": folder.as_posix(),
                "validation": metrics,
                "fit": details,
            }
            (folder / "result.json").write_text(json.dumps(candidate, indent=2), encoding="utf-8")
            candidates.append(candidate)
            print(json.dumps({k: v for k, v in candidate.items() if k != "fit"}), flush=True)
        winner = max(candidates, key=lambda c: (c["validation"]["average_precision"], -c["C"]))
        path = Path(winner["directory"]) / "validation_predictions.npz"
        with np.load(path, allow_pickle=False) as saved:
            labels, scores = saved["labels"], saved["probabilities"]
        threshold = select_joint_threshold(labels, scores, baselines)
        result = {
            "status": "complete_development_only",
            "selected_candidate": winner,
            "baselines": baselines,
            "joint_threshold": threshold,
            "joint_validation": None
            if threshold is None
            else binary_metrics(labels, scores, threshold),
            "selected_checkpoint_sha256": file_sha256(Path(winner["directory"]) / "best_model.pt"),
            "test_access": "none",
            "patient_identity_verified": False,
            "clinical_labels_verified": False,
            "full_goal_achieved": False,
            "elapsed_seconds": time.perf_counter() - start,
        }
        if any(file_sha256(Path(name)) != digest for name, digest in inputs.items()):
            raise ValueError("Registered source/input changed during experiment")
        (args.output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        status["state"] = "complete_development_only"
        save_status()
        return result
    except Exception as error:
        status.update(state="failed_preserved", error=f"{type(error).__name__}: {error}")
        save_status()
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifests", type=Path, default=Path("data/processed/ssmu_frontal_v1"))
    parser.add_argument("--output", type=Path, default=Path("models/ssmu_probe_v1"))
    parser.add_argument(
        "--feature-checkpoint",
        type=Path,
        default=Path("models/pretrained/xrv_mimic_ch/features_safe.pt"),
    )
    parser.add_argument("--original", type=Path, default=Path("models/best_model.pt"))
    parser.add_argument(
        "--current", type=Path, default=Path("models/clean_v3/selected_matched98/best_model.pt")
    )
    print(json.dumps(train(parser.parse_args()), indent=2), flush=True)


if __name__ == "__main__":
    main()
