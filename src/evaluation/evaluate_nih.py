"""Lock NIH validation selection, then compare fixed models on one fresh test cohort."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.dataset import LABEL_TO_ID
from src.data.download_nih import file_sha256
from src.data.image_preparation import CheckpointImageTransform
from src.data.nih_dataset import NIH_LABEL_TO_ID, NIHReportDataset
from src.evaluation.decision import binary_metrics, select_f1_threshold
from src.training.models import build_model
from src.training.train_clean import verify_manifests

METRICS = ("accuracy", "precision", "recall", "f1", "specificity", "roc_auc", "average_precision")


def evaluation_source_hashes() -> dict:
    """Pin the evaluator, preprocessing, labels, metrics and model construction."""
    return {
        name: file_sha256(Path(name))
        for name in (
            "src/evaluation/evaluate_nih.py",
            "src/evaluation/decision.py",
            "src/data/image_preparation.py",
            "src/data/dataset.py",
            "src/data/nih_dataset.py",
            "src/training/models.py",
        )
    }


def read_checkpoint(path: Path) -> dict:
    """Read a local checkpoint with explicit supported task and input metadata."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("label_mapping") not in (LABEL_TO_ID, NIH_LABEL_TO_ID):
        raise ValueError("Unsupported checkpoint task labels")
    if checkpoint["model_name"] == "xrv_densenet121":
        raise ValueError("NIH-pretrained XRV weights cannot provide independent NIH evaluation")
    threshold = float(checkpoint.get("decision_threshold", 0.70))
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Invalid checkpoint threshold")
    size = int(checkpoint["image_size"])
    if not 16 <= size <= 1024:
        raise ValueError("Invalid checkpoint input size")
    checkpoint["decision_threshold"] = threshold
    return checkpoint


def collect_predictions(path: Path, manifest: Path) -> tuple[np.ndarray, np.ndarray, float]:
    """Use NIH labels with each model's exact preprocessing and float32 inference."""
    checkpoint = read_checkpoint(path)
    transform = CheckpointImageTransform.from_metadata(checkpoint)
    with manifest.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if file_sha256(Path(row["filepath"])) != row["sha256"]:
                raise ValueError("Evaluation image changed after the split audit")
    dataset = NIHReportDataset(manifest, transform)
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(checkpoint["model_name"], 2, pretrained=False, freeze_backbone=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    labels, scores, loss_sum = [], [], 0.0
    with torch.inference_mode():
        for images, targets in loader:
            logits = model(images.to(device))
            if logits.shape != (len(targets), 2) or not torch.isfinite(logits).all():
                raise ValueError("Invalid model logits")
            probabilities = logits.softmax(1)[:, 1].cpu()
            loss_sum += float(nn.functional.cross_entropy(logits.cpu(), targets, reduction="sum"))
            labels.extend(targets.tolist())
            scores.extend(probabilities.tolist())
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return np.asarray(labels), np.asarray(scores), loss_sum / len(dataset)


def require_nih_audit(audit: dict) -> None:
    """Refuse target or historical-role ambiguity before any holdout scoring."""
    if (
        audit.get("class_to_idx") != NIH_LABEL_TO_ID
        or audit.get("historical_patient_roles_preserved") is not True
        or audit.get("fresh_test_exposed_patient_overlap") != 0
        or audit["cross_split_near_duplicates"]
        or any(any(values.values()) for values in audit["overlaps"].values())
    ):
        raise ValueError("NIH audit does not establish the registered fresh disjoint cohort")


def select_model(experiments: list[Path], manifests: Path, output: Path, baselines: dict) -> dict:
    """Lock both registered candidates and fixed baselines without opening the test."""
    if output.exists():
        raise FileExistsError("NIH selection already exists")
    audit = verify_manifests(manifests)
    require_nih_audit(audit)
    if set(baselines) != {"original", "current"}:
        raise ValueError("Original and current deployed baselines are required")
    if len(experiments) != 2:
        raise ValueError("Both registered NIH architecture candidates must complete")
    candidates, architectures = [], set()
    for directory in experiments:
        result = json.loads((directory / "result.json").read_bytes())
        protocol = json.loads((directory / "protocol.json").read_bytes())
        if result["status"] != "complete" or protocol["test_access"] != "none":
            raise ValueError("Candidate is unfinished or has accessed test data")
        if (
            protocol.get("task") != "nih_report_pneumonia"
            or protocol["minimum_recall"] != 0.9
            or protocol.get("feature_checkpoint") is not None
            or protocol["data_audit"]["manifest_sha256"] != audit["manifest_sha256"]
        ):
            raise ValueError("Candidate does not match the prospective NIH protocol")
        path = directory / "best_model.pt"
        checkpoint = read_checkpoint(path)
        if checkpoint["label_mapping"] != NIH_LABEL_TO_ID:
            raise ValueError("Candidate must retain NIH report-pneumonia semantics")
        checkpoint_protocol = checkpoint.get("training_protocol", {})
        if (
            checkpoint_protocol.get("task") != "nih_report_pneumonia"
            or checkpoint_protocol.get("data_audit", {}).get("manifest_sha256")
            != audit["manifest_sha256"]
            or checkpoint["model_name"] != protocol["model"]
        ):
            raise ValueError("Checkpoint provenance disagrees with its registered candidate run")
        architectures.add(checkpoint["model_name"])
        labels, probabilities, loss = collect_predictions(path, manifests / "val_manifest.csv")
        threshold = select_f1_threshold(labels, probabilities, minimum_recall=0.9)
        candidates.append(
            {
                "checkpoint": path.as_posix(),
                "sha256": file_sha256(path),
                "validation": binary_metrics(labels, probabilities, threshold),
                "validation_loss": loss,
            }
        )
    if architectures != {"resnet18", "efficientnet_v2_s"}:
        raise ValueError("Registered ResNet18 and EfficientNetV2-S candidates are required")
    winner = max(
        candidates,
        key=lambda c: (c["validation"]["average_precision"], -c["validation_loss"]),
    )
    baseline_inputs = {}
    for name, path in baselines.items():
        checkpoint = read_checkpoint(path)
        if checkpoint["label_mapping"] != LABEL_TO_ID:
            raise ValueError("Transfer baselines must be the preserved pediatric classifiers")
        baseline_inputs[name] = {
            "checkpoint": path.as_posix(),
            "sha256": file_sha256(path),
            "threshold": checkpoint["decision_threshold"],
        }
    checkpoint = read_checkpoint(Path(winner["checkpoint"]))
    checkpoint["decision_threshold"] = winner["validation"]["threshold"]
    checkpoint["validation_metrics"] = winner["validation"]
    checkpoint["validation_loss"] = winner["validation_loss"]
    checkpoint["operating_policy"] = {"minimum_validation_recall": 0.9, "criterion": "F1"}
    output.mkdir(parents=True)
    target = output / "best_model.pt"
    torch.save(checkpoint, target)
    selection = {
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_rule": "Validation average precision, then lowest unweighted loss",
        "threshold_rule": "Validation F1 under recall >=0.90",
        "test_used_for_selection": False,
        "candidates": candidates,
        "winner": winner,
        "selected_checkpoint_sha256": file_sha256(target),
        "baselines": baseline_inputs,
        "data_audit": audit,
        "evaluation_source_sha256": evaluation_source_hashes(),
    }
    (output / "selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    return selection


def paired_cluster_intervals(
    labels: np.ndarray,
    probabilities: dict[str, np.ndarray],
    thresholds: dict[str, float],
    groups: np.ndarray,
    repetitions: int = 1000,
) -> dict:
    """Resample identical patient/duplicate clusters for every model and difference."""
    if set(probabilities) != {"selected", "original", "current"}:
        raise ValueError("Paired comparison requires all three models")
    if (
        repetitions < 1
        or len(labels) != len(groups)
        or set(np.unique(labels)) != {0, 1}
        or set(thresholds) != set(probabilities)
    ):
        raise ValueError("Invalid cluster bootstrap inputs")
    for scores in probabilities.values():
        if (
            scores.shape != labels.shape
            or not np.isfinite(scores).all()
            or ((scores < 0) | (scores > 1)).any()
        ):
            raise ValueError("Bootstrap predictions are not aligned")
    rng = np.random.default_rng(20261001)
    unique = np.unique(groups)
    positions = {group: np.flatnonzero(groups == group) for group in unique}
    values = {name: {metric: [] for metric in METRICS} for name in probabilities}
    differences = {name: {metric: [] for metric in METRICS} for name in ("original", "current")}
    valid = 0
    for _ in range(repetitions):
        sampled = rng.choice(unique, len(unique), replace=True)
        indices = np.concatenate([positions[group] for group in sampled])
        if len(np.unique(labels[indices])) < 2:
            continue
        metrics = {
            name: binary_metrics(labels[indices], scores[indices], thresholds[name])
            for name, scores in probabilities.items()
        }
        valid += 1
        for name in values:
            for metric in METRICS:
                values[name][metric].append(metrics[name][metric])
        for baseline in differences:
            for metric in METRICS:
                differences[baseline][metric].append(
                    metrics["selected"][metric] - metrics[baseline][metric]
                )
    if not valid:
        raise ValueError("Bootstrap produced no replicates containing both labels")

    def summarize(samples: dict) -> dict:
        return {
            name: {metric: np.quantile(v, [0.025, 0.975]).tolist() for metric, v in rows.items()}
            for name, rows in samples.items()
        }

    return {
        "valid_repetitions": valid,
        "grouping": "Original patient IDs joined by detected image duplicates",
        "model_intervals_95": summarize(values),
        "selected_minus_baseline_intervals_95": summarize(differences),
    }


def evaluate_selected(selected: Path, manifests: Path, output: Path) -> dict:
    """Score one pinned test cohort once; never recalibrate from its labels."""
    if output.exists() or (selected / "test_started.json").exists():
        raise FileExistsError("NIH final test already started; do not silently repeat it")
    selection = json.loads((selected / "selection.json").read_bytes())
    audit = selection["data_audit"]
    require_nih_audit(audit)
    if evaluation_source_hashes() != selection["evaluation_source_sha256"]:
        raise ValueError("Evaluation implementation changed since model selection")
    paths = {"selected": selected / "best_model.pt"}
    paths.update({name: Path(row["checkpoint"]) for name, row in selection["baselines"].items()})
    hashes = {name: file_sha256(path) for name, path in paths.items()}
    if hashes["selected"] != selection["selected_checkpoint_sha256"] or any(
        hashes[name] != row["sha256"] for name, row in selection["baselines"].items()
    ):
        raise ValueError("Evaluation checkpoint changed since selection")
    manifest = manifests / "test_manifest.csv"
    if file_sha256(manifest) != audit["manifest_sha256"]["test"]:
        raise ValueError("Test cohort changed after the split audit")
    thresholds = {"selected": selection["winner"]["validation"]["threshold"]}
    thresholds.update({name: row["threshold"] for name, row in selection["baselines"].items()})
    with (selected / "test_started.json").open("x", encoding="utf-8") as stream:
        json.dump(
            {"started_at_utc": datetime.now(timezone.utc).isoformat(), "output": output.as_posix()},
            stream,
        )
    output.mkdir(parents=True)
    with manifest.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    groups = np.asarray([row["group_id"] for row in rows])
    labels, probabilities, results = None, {}, {}
    for name, path in paths.items():
        actual_labels, scores, _ = collect_predictions(path, manifest)
        if labels is not None and not np.array_equal(labels, actual_labels):
            raise ValueError("Model predictions refer to different test labels")
        labels = actual_labels
        probabilities[name] = scores
        results[name] = binary_metrics(labels, scores, thresholds[name])
        np.savez(
            output / f"{name}_predictions.npz", labels=labels, probabilities=scores, groups=groups
        )
        print(json.dumps({"model": name, "metrics": results[name]}), flush=True)
    intervals = paired_cluster_intervals(labels, probabilities, thresholds, groups)
    report = {
        "target": "Adult NIH report-derived Pneumonia versus no reported Pneumonia",
        "scope": "Same new NIH cases for every model; distinct from the old pediatric benchmark",
        "positive_prevalence": float(labels.mean()),
        "results": results,
        "paired_cluster_bootstrap": intervals,
        "checkpoint_sha256": hashes,
        "test_manifest_sha256": audit["manifest_sha256"]["test"],
        "limitations": audit["limitations"],
    }
    (output / "test_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    """Select on validation first; final test requires a separate invocation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["select", "test"])
    parser.add_argument("--experiments", type=Path, nargs="+")
    parser.add_argument("--manifests", type=Path, default=Path("data/processed/nih_pneumonia_v1"))
    parser.add_argument("--selected", type=Path, default=Path("models/nih_v1/selected"))
    parser.add_argument("--output", type=Path, default=Path("reports/metrics/nih_v1"))
    parser.add_argument("--original", type=Path, default=Path("models/best_model.pt"))
    parser.add_argument(
        "--current", type=Path, default=Path("models/clean_v3/selected_matched98/best_model.pt")
    )
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.action == "select":
        if not args.experiments:
            parser.error("Selection requires completed experiments")
        result = select_model(
            args.experiments,
            args.manifests,
            args.selected,
            {"original": args.original, "current": args.current},
        )
    else:
        result = evaluate_selected(args.selected, args.manifests, args.output)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
