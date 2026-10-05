"""Lock validation-only model selection before opening the legacy test benchmark."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from src.data.dataset import ChestXRayDataset
from src.evaluation.decision import binary_metrics, select_threshold
from src.inference.predict import load_model, preprocess_image
from src.training.train_clean import verify_manifests


def collect_predictions(checkpoint_path: Path, manifest: Path) -> tuple[np.ndarray, np.ndarray]:
    """Use the deployed model's preprocessing and full-precision inference."""
    loaded = load_model(checkpoint_path)
    with manifest.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            source = Path(row["filepath"])
            if hashlib.sha256(source.read_bytes()).hexdigest() != row["sha256"]:
                raise ValueError(f"Image changed after audit: {source}")
    # Read metadata only through Dataset; decode using exactly the serving transform.
    dataset = ChestXRayDataset(manifest)
    labels, scores = [], []
    with torch.inference_mode():
        for start in range(0, len(dataset), 24):
            batch = dataset.samples[start : start + 24]
            images = torch.cat(
                [
                    preprocess_image(
                        sample.filepath,
                        loaded.image_size,
                        loaded.normalization_mean,
                        loaded.normalization_std,
                        preprocessing=loaded.preprocessing,
                    )
                    for sample in batch
                ]
            )
            probabilities = loaded.model(images.to(loaded.device)).softmax(1)[:, 1].cpu().numpy()
            labels.extend(sample.label_id for sample in batch)
            scores.extend(probabilities.tolist())
    return np.asarray(labels), np.asarray(scores)


def select_model(
    experiments: list[Path], manifests: Path, output: Path, minimum_recall: float | None = None
) -> dict:
    """Compare completed candidates on validation, lock winner, and save its threshold."""
    if output.exists():
        raise FileExistsError(f"Selection already exists: {output}")
    audit = verify_manifests(manifests)
    candidates = []
    recall_floor = minimum_recall
    for directory in experiments:
        result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
        if result["status"] != "complete":
            raise ValueError(f"Unfinished experiment: {directory}")
        protocol = json.loads((directory / "protocol.json").read_text(encoding="utf-8"))
        if protocol["data_audit"]["manifest_sha256"] != audit["manifest_sha256"]:
            raise ValueError("Experiments use different data splits")
        if recall_floor is None:
            recall_floor = float(protocol["minimum_recall"])
        elif minimum_recall is None and recall_floor != float(protocol["minimum_recall"]):
            raise ValueError(
                "Candidate policies differ; specify one shared validation recall floor"
            )
        checkpoint_path = directory / "best_model.pt"
        labels, scores = collect_predictions(checkpoint_path, manifests / "val_manifest.csv")
        threshold = select_threshold(labels, scores, recall_floor)
        metrics = binary_metrics(labels, scores, threshold)
        candidates.append(
            {
                "checkpoint": checkpoint_path.as_posix(),
                "validation": metrics,
                "sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
            }
        )
    # Prespecified selection: specificity at the common recall floor; F1 breaks ties.
    winner = max(candidates, key=lambda c: (c["validation"]["specificity"], c["validation"]["f1"]))
    checkpoint = torch.load(winner["checkpoint"], map_location="cpu", weights_only=False)
    checkpoint["decision_threshold"] = winner["validation"]["threshold"]
    checkpoint["validation_metrics"] = winner["validation"]
    checkpoint["operating_policy"] = {"minimum_validation_recall": recall_floor}
    output.mkdir(parents=True)
    target = output / "best_model.pt"
    torch.save(checkpoint, target)
    selection = {
        "selection_rule": f"validation specificity at recall>={recall_floor}, then F1",
        "minimum_validation_recall": recall_floor,
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidates": candidates,
        "winner": winner,
        "selected_checkpoint_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "data_audit": audit,
        "test_used_for_selection": False,
    }
    (output / "selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    return selection


def bootstrap_intervals(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    groups: np.ndarray,
    repetitions: int = 1000,
) -> dict:
    """Patient-proxy cluster bootstrap; correlated images stay in a sampled cluster."""
    rng = np.random.default_rng(20260915)
    unique_groups = np.unique(groups)
    indices = {group: np.flatnonzero(groups == group) for group in unique_groups}
    values = {
        name: [] for name in ("accuracy", "precision", "recall", "specificity", "f1", "roc_auc")
    }
    for _ in range(repetitions):
        sampled = rng.choice(unique_groups, len(unique_groups), replace=True)
        positions = np.concatenate([indices[group] for group in sampled])
        if len(np.unique(labels[positions])) < 2:
            continue
        metrics = binary_metrics(labels[positions], scores[positions], threshold)
        for name in values:
            values[name].append(metrics[name])
    return {name: np.quantile(samples, [0.025, 0.975]).tolist() for name, samples in values.items()}


def evaluate_selected(output: Path, manifests: Path) -> dict:
    """Evaluate only the locked winner; never tune on the resulting test metrics."""
    selection = json.loads((output / "selection.json").read_text(encoding="utf-8"))
    target = output / "best_model.pt"
    if hashlib.sha256(target.read_bytes()).hexdigest() != selection["selected_checkpoint_sha256"]:
        raise ValueError("Selected checkpoint changed after model selection")
    manifest = manifests / "test_manifest.csv"
    expected = selection["data_audit"]["manifest_sha256"]["test"]
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != expected:
        raise ValueError("Test manifest changed since audit")
    if (output / "test_metrics.json").exists():
        raise FileExistsError("Final test report already exists; do not silently repeat selection")
    labels, scores = collect_predictions(target, manifest)
    threshold = selection["winner"]["validation"]["threshold"]
    with manifest.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    groups = np.asarray([row["group_id"] for row in rows])
    report = {
        "test_metrics": binary_metrics(labels, scores, threshold),
        "confidence_intervals_95": bootstrap_intervals(labels, scores, threshold, groups),
        "checkpoint_sha256": selection["selected_checkpoint_sha256"],
        "test_manifest_sha256": expected,
        "benchmark_status": "Legacy test previously inspected by earlier project versions",
        "grouping_limitations": selection["data_audit"]["limitations"],
    }
    np.savez(output / "test_predictions.npz", labels=labels, probabilities=scores, groups=groups)
    (output / "test_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    """Select first, inspect test only in a separate explicit invocation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["select", "test"])
    parser.add_argument("--experiments", type=Path, nargs="+")
    parser.add_argument("--manifests", type=Path, default=Path("data/processed/clean_v1"))
    parser.add_argument("--output", type=Path, default=Path("models/clean_v1/selected"))
    parser.add_argument("--minimum-recall", type=float)
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.action == "select":
        if not args.experiments:
            parser.error("select requires --experiments")
        result = select_model(args.experiments, args.manifests, args.output, args.minimum_recall)
    else:
        result = evaluate_selected(args.output, args.manifests)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
