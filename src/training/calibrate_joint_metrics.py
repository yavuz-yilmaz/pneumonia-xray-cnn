"""Calibrate a fixed learned model against both baselines using validation only."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

from src.data.download_nih import file_sha256
from src.evaluation.decision import binary_metrics
from src.evaluation.evaluate_nih import collect_predictions
from src.evaluation.evaluate_openi import model_inputs, verify_hashes

JOINT_METRICS = ("accuracy", "precision", "recall", "f1")


def select_joint_threshold(labels: np.ndarray, scores: np.ndarray, baselines: dict) -> float | None:
    """Find an inclusive threshold dominating both validation references."""
    if (
        labels.ndim != 1
        or scores.shape != labels.shape
        or set(np.unique(labels)) != {0, 1}
        or not np.isfinite(scores).all()
        or ((scores < 0) | (scores > 1)).any()
        or set(baselines) != {"original", "current"}
    ):
        raise ValueError("Invalid joint validation inputs")
    floors = {key: max(float(baselines[name][key]) for name in baselines) for key in JOINT_METRICS}
    if any(not np.isfinite(value) or not 0 <= value <= 1 for value in floors.values()):
        raise ValueError("Invalid baseline metric")
    order = np.argsort(-scores, kind="stable")
    ordered_scores, ordered_labels = scores[order], labels[order]
    ends = np.flatnonzero(np.r_[ordered_scores[:-1] != ordered_scores[1:], True])
    tp = np.cumsum(ordered_labels)[ends].astype(float)
    fp = (ends + 1) - tp
    positives = float(labels.sum())
    negatives = len(labels) - positives
    values = {
        "accuracy": (tp + negatives - fp) / len(labels),
        "precision": tp / (tp + fp),
        "recall": tp / positives,
        "f1": 2 * tp / (positives + tp + fp),
    }
    feasible = np.ones(len(ends), dtype=bool)
    improved = np.zeros(len(ends), dtype=bool)
    for key in JOINT_METRICS:
        feasible &= values[key] >= floors[key] - 1e-12
        improved |= values[key] > floors[key] + 1e-12
    positions = np.flatnonzero(feasible & improved)
    if not len(positions):
        return None
    winner = max(
        positions,
        key=lambda index: (
            values["f1"][index],
            values["accuracy"][index],
            ordered_scores[ends[index]],
        ),
    )
    return float(ordered_scores[ends[winner]])


def calibrate(root: Path) -> dict:
    """Preserve the primary head; score fixed baselines on the NIH validation set."""
    run = root / "reports/metrics/joint_metrics_run_v1"
    registration = json.loads((run / "registration.json").read_bytes())
    verify_hashes(root, registration["source_and_cohort_sha256"])
    output = root / "models/nih_mimic_v1/joint_policy"
    if output.exists():
        raise FileExistsError("Joint-policy output already exists; preserving it")
    records = model_inputs(root)
    primary_path = Path(records["mimic"]["path"])
    manifest = root / "data/processed/nih_pneumonia_v1/val_manifest.csv"
    with manifest.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    expected_labels = np.asarray([int(row["label_id"]) for row in rows])
    with np.load(primary_path.with_name("validation_predictions.npz"), allow_pickle=False) as saved:
        labels, scores = saved["labels"], saved["probabilities"]
    if not np.array_equal(labels, expected_labels):
        raise ValueError("Primary validation prediction alignment failed")
    torch.set_num_threads(4)
    output.mkdir(parents=True)
    baselines = {}
    for name in ("original", "current"):
        path = Path(records[name]["path"])
        if file_sha256(path) != records[name]["sha256"]:
            raise ValueError("Fixed baseline changed")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        targets, probabilities, _ = collect_predictions(path, manifest)
        if not np.array_equal(targets, labels):
            raise ValueError("Baseline validation label order differs")
        threshold = float(checkpoint.get("decision_threshold", 0.70))
        baselines[name] = binary_metrics(labels, probabilities, threshold)
        np.savez(
            output / f"{name}_validation_predictions.npz",
            labels=labels,
            probabilities=probabilities,
        )
        print(json.dumps({"baseline_validation": name, "metrics": baselines[name]}), flush=True)
    threshold = select_joint_threshold(labels, scores, baselines)
    summary = {
        "status": "no_feasible_validation_threshold" if threshold is None else "complete",
        "baselines": baselines,
        "validation_manifest_sha256": file_sha256(manifest),
        "primary_checkpoint_sha256": records["mimic"]["sha256"],
        "primary_validation_predictions_sha256": file_sha256(
            primary_path.with_name("validation_predictions.npz")
        ),
        "test_access": "none",
        "openi_access": "none",
        "new_head_fitting": False,
        "protocol": "docs/nih_joint_metrics_protocol.md",
    }
    if threshold is not None:
        checkpoint = torch.load(primary_path, map_location="cpu", weights_only=True)
        metrics = binary_metrics(labels, scores, threshold)
        checkpoint["decision_threshold"] = threshold
        checkpoint["validation_metrics"] = metrics
        checkpoint["model_version"] = "nih_mimic_joint_validation_policy"
        checkpoint["joint_validation_policy"] = dict(summary, validation=metrics)
        torch.save(checkpoint, output / "best_model.pt")
        np.savez(output / "validation_predictions.npz", labels=labels, probabilities=scores)
        summary.update(validation=metrics, checkpoint_sha256=file_sha256(output / "best_model.pt"))
    (output / "result.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    """Run the prospectively specified validation-only calibration."""
    print(json.dumps(calibrate(Path.cwd()), indent=2), flush=True)


if __name__ == "__main__":
    main()
