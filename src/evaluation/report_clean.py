"""Publish the locked experiment's measured comparison without selecting new models."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.evaluation.decision import binary_metrics


def write_report(selected: Path, destination: Path) -> None:
    """Compare the same ordered legacy test images and render aggregate figures."""
    selection = json.loads((selected / "selection.json").read_text(encoding="utf-8"))
    report = json.loads((selected / "test_metrics.json").read_text(encoding="utf-8"))
    predictions = np.load(selected / "test_predictions.npz")
    previous = json.loads(
        Path("reports/metrics/legacy_v0_test_metrics.json").read_text(encoding="utf-8")
    )
    old_labels = np.array([entry["label_id"] for entry in previous["predictions"]])
    old_scores = np.array([entry["pneumonia_probability"] for entry in previous["predictions"]])
    manifests = [
        Path("data/processed/test_manifest.csv"),
        Path("data/processed/clean_v1/test_manifest.csv"),
    ]
    paths = []
    for manifest in manifests:
        with manifest.open(encoding="utf-8", newline="") as stream:
            paths.append([Path(row["filepath"]).resolve() for row in csv.DictReader(stream)])
    if paths[0] != paths[1] or not np.array_equal(old_labels, predictions["labels"]):
        raise ValueError("Old and new predictions do not represent the same ordered test cases")
    old_metrics = binary_metrics(old_labels, old_scores, 0.7)
    new_metrics = report["test_metrics"]
    names = {
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "specificity": "Specificity",
        "f1": "F1",
        "roc_auc": "ROC-AUC",
    }
    rows = []
    for key, label in names.items():
        low, high = report["confidence_intervals_95"][key]
        delta = (new_metrics[key] - old_metrics[key]) * 100
        rows.append(
            f"| {label} | {old_metrics[key]:.2%} | {new_metrics[key]:.2%} | "
            f"{delta:+.2f} | {low:.2%}–{high:.2%} |"
        )
    old_matrix = np.array(old_metrics["confusion_matrix"])
    new_matrix = np.array(new_metrics["confusion_matrix"])
    figure_dir = destination.parent / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    figure_path = figure_dir / f"{destination.stem}_comparison.png"
    figure, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    for axis, matrix, title in zip(
        axes,
        (old_matrix, new_matrix),
        ("Previous model (threshold 0.70)", "Clean selected model"),
        strict=True,
    ):
        axis.imshow(matrix, cmap="Blues", vmin=0, vmax=max(old_matrix.max(), new_matrix.max()))
        for row in range(2):
            for column in range(2):
                axis.text(
                    column,
                    row,
                    str(matrix[row, column]),
                    ha="center",
                    va="center",
                    color="white" if matrix[row, column] > 200 else "black",
                    fontsize=15,
                )
        axis.set_xticks([0, 1], ["NORMAL", "PNEUMONIA"])
        axis.set_yticks([0, 1], ["NORMAL", "PNEUMONIA"])
        axis.set(xlabel="Predicted", ylabel="Actual", title=title)
    figure.savefig(figure_path, dpi=140)
    plt.close(figure)
    comparison = {
        "previous_threshold": 0.7,
        "previous": old_metrics,
        "new": new_metrics,
        "delta_percentage_points": {
            key: 100 * (new_metrics[key] - old_metrics[key]) for key in names
        },
    }
    (selected / "comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    winner = selection["winner"]
    validation = winner["validation"]
    audit = selection["data_audit"]
    recall_floor = float(selection.get("minimum_validation_recall", 0.99))
    counts = audit["counts"]
    candidate_rows = "\n".join(
        f"| {Path(c['checkpoint']).parent.name} | {c['validation']['accuracy']:.2%} | "
        f"{c['validation']['precision']:.2%} | {c['validation']['recall']:.2%} | "
        f"{c['validation']['f1']:.2%} | {c['validation']['specificity']:.2%} |"
        for c in selection["candidates"]
    )
    text = f"""# Clean retraining: measured model comparison

## Locked selection

- Selected experiment: `{Path(winner["checkpoint"]).parent.name}`
- Candidate checkpoint (not automatically deployed): `{(selected / "best_model.pt").as_posix()}`
- Checkpoint SHA-256: `{selection["selected_checkpoint_sha256"]}`
- Operating threshold: `{new_metrics["threshold"]:.8f}`
- Selection used validation specificity at recall >={recall_floor:.0%}; F1 breaks ties.
- The previous deployed model's documented validation recall floor was 98%.
- The threshold was fixed before evaluating the selected model on the legacy test.
- Validation operating point: recall {validation["recall"]:.2%},
  specificity {validation["specificity"]:.2%}, F1 {validation["f1"]:.2%}.

## Same 624-image legacy test

The previous model uses its deployed 0.70 threshold. New metrics use the new
validation-selected threshold. Neither threshold is optimized on these test labels.
Changes are percentage points; intervals are 95% patient-proxy cluster bootstrap
intervals (1,000 resamples), not evidence of clinical reliability.

| Metric | Previous | Clean selected | Change (pp) | New 95% interval |
|---|---:|---:|---:|---:|
{chr(10).join(rows)}

- False positives: **{old_matrix[0, 1]} → {new_matrix[0, 1]}**.
- False negatives: **{old_matrix[1, 0]} → {new_matrix[1, 0]}**.

![Same-test confusion matrices](figures/{figure_path.name})

## Validation comparison used for selection

These validation results must not be compared directly with the test results above.

| Experiment | Accuracy | Precision | Recall | F1 | Specificity |
|---|---:|---:|---:|---:|---:|
{candidate_rows}

## Leakage controls and limits

- Train: {counts["train"]["images"]} images; validation: {counts["val"]["images"]};
  unchanged original test: {counts["test"]["images"]}.
- Excluded development images: {counts["excluded"]["images"]}; raw files retained.
- No shared filename-derived patient groups, byte hashes, decoded pixel hashes,
  or connected duplicate groups between any pair of splits.
- Detected cross-split near-duplicate pairs: {audit["cross_split_near_duplicates"]}.
- Identities are conservative filename proxies. Actual patient-disjointness cannot
  be established beyond the supplied metadata. Similarity screening is not exhaustive.
- The legacy test was inspected by older project versions. This is a same-dataset
  benchmark comparison, not untouched external validation or clinical validation.
- Validation threshold selection has its own estimation uncertainty; the {recall_floor:.0%} recall
  target does not guarantee the same recall on a new population.

## Reproducibility

See [the experiment protocol](clean_training_protocol.md). Full audit, excluded
manifest and original source hashes are in `data/processed/clean_v1`. Each experiment
contains its configuration, training history, validation predictions, and checkpoint.
The selected directory contains the locked selection, test predictions, bootstrap
intervals, and the machine-readable comparison. Previous artifacts are retained.

Educational/research model; not a medical diagnostic device.
"""
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text, encoding="utf-8")


def main() -> None:
    """Generate a report only after the locked test evaluation exists."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected", type=Path, default=Path("models/clean_v1/selected"))
    parser.add_argument("--destination", type=Path, default=Path("docs/clean_model_results.md"))
    args = parser.parse_args()
    write_report(args.selected, args.destination)


if __name__ == "__main__":
    main()
