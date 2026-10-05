"""Check a frozen SSMU classifier on previously observed OpenI cases."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from src.data.dataset import LABEL_TO_ID
from src.data.download_nih import file_sha256
from src.evaluation.evaluate_openi import (
    METRICS,
    aggregate_cases,
    case_metrics,
    predict_images,
    verify_hashes,
)
from src.training.train_xray_probe import MIMIC_FEATURE_SHA256

MODEL_SHA256 = "6e8202040581f8e8412fea206f66e1eb748471e0161fff3d4cc56396b237fbc6"
THRESHOLD = 0.7265916466712952
NAMES = ("ssmu", "original", "current")


def paired_intervals(
    labels: np.ndarray, scores: dict, thresholds: dict, repetitions: int = 1000
) -> dict:
    """Resample the same cases, retaining undefined precision and valid counts."""
    if set(scores) != set(NAMES) or set(thresholds) != set(NAMES):
        raise ValueError("Require the selected model and both baselines")
    if repetitions < 1 or set(np.unique(labels)) != {0, 1}:
        raise ValueError("Require both classes and positive repetitions")
    for name in NAMES:
        values = scores[name]
        if (
            values.shape != labels.shape
            or not np.isfinite(values).all()
            or ((values < 0) | (values > 1)).any()
            or not np.isfinite(thresholds[name])
            or not 0 <= thresholds[name] <= 1
        ):
            raise ValueError("Invalid probabilities or threshold")
    draws = {name: {metric: [] for metric in METRICS} for name in NAMES}
    differences = {name: {metric: [] for metric in METRICS} for name in NAMES[1:]}
    rng = np.random.default_rng(20261002)
    valid = 0
    for _ in range(repetitions):
        positions = rng.choice(len(labels), len(labels), replace=True)
        if len(np.unique(labels[positions])) < 2:
            continue
        valid += 1
        results = {
            name: case_metrics(labels[positions], scores[name][positions], thresholds[name])
            for name in NAMES
        }
        for name in NAMES:
            for metric in METRICS:
                value = results[name][metric]
                candidate = results["ssmu"][metric]
                if value is not None:
                    draws[name][metric].append(value)
                    if name != "ssmu" and candidate is not None:
                        differences[name][metric].append(candidate - value)

    def summarize(items: dict) -> dict:
        return {
            name: {
                metric: {
                    "valid_repetitions": len(values),
                    "interval_95": np.percentile(values, [2.5, 97.5]).tolist() if values else None,
                }
                for metric, values in per_metric.items()
            }
            for name, per_metric in items.items()
        }

    return {
        "seed": 20261002,
        "unit": "OpenI case; original patient identities unavailable",
        "requested_repetitions": repetitions,
        "both_class_repetitions": valid,
        "models": summarize(draws),
        "ssmu_minus_comparator": summarize(differences),
    }


def evaluate(root: Path, output: Path) -> dict:
    """Register once, reuse checked baseline scores, then infer only the SSMU model."""
    if output.exists():
        raise FileExistsError("Exploratory evaluation already exists; preserving it")
    primary = root / "reports/metrics/openi_external_v1"
    secondary = root / "reports/metrics/openi_source_head_v1"
    for directory in (primary, secondary):
        if json.loads((directory / "status.json").read_bytes())["state"] != "complete":
            raise ValueError("Previous OpenI studies must be complete")
    previous = json.loads((secondary / "registration.json").read_bytes())
    verify_hashes(root, previous["source_sha256"])
    verify_hashes(root, previous["primary_prediction_sha256"])
    cohort = root / "data/processed/openi_external_v1"
    if file_sha256(cohort / "cohort_audit.json") != previous["cohort_audit_sha256"]:
        raise ValueError("Original cohort audit changed")
    audit = json.loads((cohort / "cohort_audit.json").read_bytes())
    verify_hashes(cohort, audit["artifact_sha256"])
    verify_hashes(root, audit["input_sha256"])
    with (cohort / "manifest.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != audit["image_count"]:
        raise ValueError("Cohort image count differs")
    for row in rows:
        if file_sha256(Path(row["filepath"])) != row["sha256"]:
            raise ValueError("Retained image changed")
    image_labels = np.asarray([int(row["label_id"]) for row in rows])
    ids, labels, _ = aggregate_cases(rows, image_labels, np.zeros(len(rows)))
    if len(ids) != audit["case_count"]:
        raise ValueError("Cohort case count differs")
    primary_result = json.loads((primary / "result.json").read_bytes())
    scores, thresholds = {}, {"ssmu": THRESHOLD}
    for name in NAMES[1:]:
        thresholds[name] = previous["thresholds"][name]
        with np.load(primary / f"{name}_predictions.npz", allow_pickle=False) as saved:
            saved_ids, targets, aggregated = aggregate_cases(
                rows, image_labels, saved["image_probabilities"]
            )
            if (
                saved_ids != ids
                or not np.array_equal(saved["case_ids"], ids)
                or not np.array_equal(saved["labels"], labels)
                or not np.array_equal(targets, labels)
                or not np.array_equal(saved["probabilities"], aggregated)
                or case_metrics(labels, aggregated, thresholds[name])
                != primary_result["models"][name]
            ):
                raise ValueError("Saved baseline alignment, aggregation or metrics differ")
            scores[name] = aggregated
    model_path = root / "models/ssmu_probe_v1/selected_joint/best_model.pt"
    if file_sha256(model_path) != MODEL_SHA256:
        raise ValueError("Frozen SSMU model changed")
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    if (
        checkpoint["decision_threshold"] != THRESHOLD
        or checkpoint["label_mapping"] != LABEL_TO_ID
        or checkpoint["training_protocol"]["task"] != "ssmu_source_pneumonia"
        or checkpoint["training_protocol"]["test_access"] != "none"
    ):
        raise ValueError("Frozen SSMU decision or training provenance changed")
    feature_path = root / "models/pretrained/xrv_mimic_ch/features_safe.pt"
    if file_sha256(feature_path) != MIMIC_FEATURE_SHA256:
        raise ValueError("Verified MIMIC-only features changed")
    features = torch.load(feature_path, map_location="cpu", weights_only=True)["feature_state_dict"]
    state = checkpoint["model_state_dict"]
    if any(not torch.isfinite(t).all() for t in state.values()) or any(
        not torch.equal(state[f"features.{name}"], tensor) for name, tensor in features.items()
    ):
        raise ValueError("Frozen feature tensors differ or are nonfinite")
    sources = dict(previous["source_sha256"])
    for name in (
        "src/evaluation/evaluate_ssmu_openi.py",
        "docs/ssmu_openi_check_protocol.md",
    ):
        sources[name] = file_sha256(root / name)
    registration = {
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "classification": "exploratory; OpenI previously observed for five models",
        "model_sha256": MODEL_SHA256,
        "thresholds": thresholds,
        "source_sha256": sources,
        "baseline_prediction_sha256": previous["primary_prediction_sha256"],
        "cohort_audit_sha256": previous["cohort_audit_sha256"],
        "primary_result_sha256": file_sha256(primary / "result.json"),
        "ssmu_selection_sha256": file_sha256(model_path.with_name("selection.json")),
        "cohort": audit,
        "no_openi_fitting_or_tuning": True,
        "automatic_deployment": False,
    }
    output.mkdir(parents=True)
    (output / "registration.json").write_text(json.dumps(registration, indent=2), encoding="utf-8")

    def status(value: dict) -> None:
        temporary = output / "status.tmp"
        temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
        temporary.replace(output / "status.json")

    status({"state": "started"})
    try:
        torch.set_num_threads(4)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        targets, probabilities = predict_images(checkpoint, cohort / "manifest.csv", device)
        new_ids, new_labels, scores["ssmu"] = aggregate_cases(rows, targets, probabilities)
        if new_ids != ids or not np.array_equal(new_labels, labels):
            raise ValueError("New predictions are not aligned to saved baseline cases")
        np.savez(
            output / "ssmu_predictions.npz",
            case_ids=np.asarray(ids),
            labels=labels,
            probabilities=scores["ssmu"],
            image_probabilities=probabilities,
        )
        results = {name: case_metrics(labels, scores[name], thresholds[name]) for name in NAMES}
        print(json.dumps({"models": results}), flush=True)
        report = {
            "status": "complete_exploratory",
            "models": results,
            "case_count": len(labels),
            "positive_cases": int(labels.sum()),
            "target": audit["target"],
            "unit": audit["unit"],
            "paired_intervals": paired_intervals(labels, scores, thresholds),
            "untouched_final_test": False,
            "no_threshold_or_model_tuning": True,
            "no_automatic_deployment": True,
        }
        (output / "result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        status({"state": "complete_exploratory"})
        return report
    except Exception as error:
        status({"state": "failed_preserved", "error": f"{type(error).__name__}: {error}"})
        raise


if __name__ == "__main__":
    evaluate(Path.cwd(), Path("reports/metrics/ssmu_openi_check_v1"))
