"""Score the prospectively specified source head once and retain all five outcomes."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from src.data.download_nih import file_sha256
from src.data.nih_dataset import NIH_LABEL_TO_ID
from src.evaluation.evaluate_openi import (
    METRICS,
    MODEL_NAMES,
    SOURCE_FILES,
    aggregate_cases,
    case_metrics,
    predict_images,
    verify_hashes,
)
from src.training.export_mimic_head import load_source_head
from src.training.train_xray_probe import MIMIC_FEATURE_SHA256


def paired_differences(
    labels: np.ndarray, scores: dict, thresholds: dict, repetitions: int = 1000
) -> dict:
    """Compare the source head with every fixed comparator on identical resamples."""
    if set(scores) != MODEL_NAMES | {"source_head"} or set(scores) != set(thresholds):
        raise ValueError("All five fixed models are required")
    if repetitions < 1 or set(np.unique(labels)) != {0, 1}:
        raise ValueError("Invalid paired evaluation cohort")
    for name, values in scores.items():
        if (
            values.shape != labels.shape
            or not np.isfinite(values).all()
            or ((values < 0) | (values > 1)).any()
        ):
            raise ValueError("Invalid probabilities")
        if not np.isfinite(thresholds[name]) or not 0 <= thresholds[name] <= 1:
            raise ValueError("Invalid fixed threshold")
    draws = {name: {metric: [] for metric in METRICS} for name in MODEL_NAMES}
    rng = np.random.default_rng(20261001)
    valid = 0
    for _ in range(repetitions):
        positions = rng.choice(len(labels), len(labels), replace=True)
        if len(np.unique(labels[positions])) < 2:
            continue
        valid += 1
        candidate = case_metrics(
            labels[positions], scores["source_head"][positions], thresholds["source_head"]
        )
        for name in MODEL_NAMES:
            comparator = case_metrics(labels[positions], scores[name][positions], thresholds[name])
            for metric in METRICS:
                if candidate[metric] is not None and comparator[metric] is not None:
                    draws[name][metric].append(candidate[metric] - comparator[metric])
    return {
        "seed": 20261001,
        "requested_repetitions": repetitions,
        "both_class_repetitions": valid,
        "source_head_minus_comparator": {
            name: {
                metric: {
                    "valid_repetitions": len(values),
                    "interval_95": np.percentile(values, [2.5, 97.5]).tolist() if values else None,
                }
                for metric, values in per_metric.items()
            }
            for name, per_metric in draws.items()
        },
    }


def read_source_checkpoint(root: Path) -> tuple[dict, str]:
    """Verify every copied source feature and classifier parameter before scoring."""
    run = root / "reports/metrics/nih_mimic_source_head_run_v1"
    if json.loads((run / "status.json").read_bytes())["state"] != "complete_source_validation_only":
        raise ValueError("Source-head calibration is incomplete")
    registration = json.loads((run / "registration.json").read_bytes())
    verify_hashes(root, registration["source_and_cohort_sha256"])
    directory = root / "models/nih_mimic_v1/source_head"
    result = json.loads((directory / "result.json").read_bytes())
    checkpoint_path = directory / "best_model.pt"
    digest = file_sha256(checkpoint_path)
    if result["status"] != "complete" or digest != result["checkpoint_sha256"]:
        raise ValueError("Source checkpoint changed or export incomplete")
    protocol = json.loads((directory / "protocol.json").read_bytes())
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if (
        checkpoint["model_name"] != "xrv_densenet121"
        or checkpoint["label_mapping"] != NIH_LABEL_TO_ID
        or checkpoint["validation_metrics"] != result["validation"]
        or json.dumps(checkpoint["training_protocol"], sort_keys=True)
        != json.dumps(protocol, sort_keys=True)
        or protocol["head_training_on_nih"] != "none"
        or protocol["test_access"] != "none"
        or protocol["openi_access"] != "none"
        or protocol["minimum_recall"] != 0.9
        or result["validation"]["recall"] < 0.9
        or checkpoint["decision_threshold"] != result["validation"]["threshold"]
    ):
        raise ValueError("Source checkpoint violates the fixed calibration contract")
    head = load_source_head(root / "models/pretrained/xrv_mimic_ch/pneumonia_head_safe.pt")
    feature_path = root / "models/pretrained/xrv_mimic_ch/features_safe.pt"
    if file_sha256(feature_path) != MIMIC_FEATURE_SHA256:
        raise ValueError("Approved MIMIC features changed")
    features = torch.load(feature_path, map_location="cpu", weights_only=True)["feature_state_dict"]
    state = checkpoint["model_state_dict"]
    if (
        set(state) != {"classifier.weight", "classifier.bias"} | {"features." + k for k in features}
        or any(not torch.equal(state["features." + k], value) for k, value in features.items())
        or not torch.equal(state["classifier.weight"][1], head["weight"])
        or not torch.equal(state["classifier.bias"][1], head["bias"])
        or state["classifier.weight"][0].count_nonzero()
        or state["classifier.bias"][0] != 0
    ):
        raise ValueError("Exported parameters differ from the approved source model")
    if (
        checkpoint["image_size"] != 224
        or checkpoint["preprocessing"] != "resize"
        or tuple(checkpoint["normalization"]["mean"]) != (0.5, 0.5, 0.5)
        or tuple(checkpoint["normalization"]["std"]) != (1 / 2048,) * 3
    ):
        raise ValueError("Source input semantics changed")
    return checkpoint, digest


def evaluate(root: Path, output: Path, device_name: str = "auto") -> dict:
    """Reuse four saved case predictions; infer only the fixed secondary source head."""
    if output.exists():
        raise FileExistsError("Secondary evaluation already exists; preserving it")
    primary = root / "reports/metrics/openi_external_v1"
    if json.loads((primary / "status.json").read_bytes())["state"] != "complete":
        raise ValueError("The primary comparison must finish first")
    cohort = root / "data/processed/openi_external_v1"
    audit = json.loads((cohort / "cohort_audit.json").read_bytes())
    verify_hashes(cohort, audit["artifact_sha256"])
    verify_hashes(root, audit["source_sha256"])
    verify_hashes(root, audit["input_sha256"])
    with (cohort / "manifest.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        if file_sha256(Path(row["filepath"])) != row["sha256"]:
            raise ValueError("Evaluation image changed")
    checkpoint, digest = read_source_checkpoint(root)
    primary_registration = json.loads((primary / "registration.json").read_bytes())
    if primary_registration["cohort_audit_sha256"] != file_sha256(cohort / "cohort_audit.json"):
        raise ValueError("Primary and secondary cohorts differ")
    expected_ids = sorted({r["case_id"] for r in rows})
    expected_labels = {r["case_id"]: int(r["label_id"]) for r in rows}
    labels = np.array([expected_labels[k] for k in expected_ids])
    scores, primary_hashes = {}, {}
    for name in sorted(MODEL_NAMES):
        path = primary / f"{name}_predictions.npz"
        primary_hashes[path.relative_to(root).as_posix()] = file_sha256(path)
        with np.load(path, allow_pickle=False) as saved:
            if saved["case_ids"].tolist() != expected_ids or not np.array_equal(
                saved["labels"], labels
            ):
                raise ValueError("Saved comparator case alignment failed")
            scores[name] = saved["probabilities"]
    thresholds = dict(
        primary_registration["thresholds"], source_head=checkpoint["decision_threshold"]
    )
    files = (
        *SOURCE_FILES,
        "src/evaluation/evaluate_mimic_source_head.py",
        "src/training/export_mimic_head.py",
        "docs/nih_mimic_source_head_protocol.md",
    )
    sources = {name: file_sha256(root / name) for name in files}
    registration = {
        "source_sha256": sources,
        "source_checkpoint_sha256": digest,
        "source_calibration_result_sha256": file_sha256(
            root / "models/nih_mimic_v1/source_head/result.json"
        ),
        "primary_prediction_sha256": primary_hashes,
        "thresholds": thresholds,
        "cohort_audit_sha256": file_sha256(cohort / "cohort_audit.json"),
        "primary_registration_sha256": file_sha256(primary / "registration.json"),
        "prospective_recipe": "docs/nih_mimic_source_head_protocol.md",
        "no_candidate_or_threshold_selection_from_openi": True,
        "primary_comparison_preserved": True,
    }
    output.mkdir(parents=True)
    (output / "registration.json").write_text(json.dumps(registration, indent=2), encoding="utf-8")
    status = {"state": "started"}

    def save_status() -> None:
        temporary = output / "status.tmp"
        temporary.write_text(json.dumps(status, indent=2), encoding="utf-8")
        temporary.replace(output / "status.json")

    save_status()
    try:
        torch.set_num_threads(4)
        device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device_name == "auto"
            else torch.device(device_name)
        )
        image_labels, image_scores = predict_images(checkpoint, cohort / "manifest.csv", device)
        ids, targets, source_scores = aggregate_cases(rows, image_labels, image_scores)
        if ids != expected_ids or not np.array_equal(targets, labels):
            raise ValueError("Source head case alignment failed")
        scores["source_head"] = source_scores
        metrics = {
            name: case_metrics(labels, values, thresholds[name]) for name, values in scores.items()
        }
        existing = json.loads((primary / "result.json").read_bytes())
        if any(metrics[name] != existing["models"][name] for name in MODEL_NAMES):
            raise ValueError("Saved comparator predictions disagree with primary report")
        np.savez(
            output / "source_head_predictions.npz",
            case_ids=np.array(ids),
            labels=labels,
            probabilities=source_scores,
            image_probabilities=image_scores,
        )
        report = {
            "status": "complete",
            "models": metrics,
            "paired_intervals": paired_differences(labels, scores, thresholds),
            "case_count": len(labels),
            "positive_cases": int(np.count_nonzero(labels)),
            "target": audit["target"],
            "unit": audit["unit"],
            "no_automatic_deployment": True,
            "primary_candidates_not_replaced": True,
            "interpretation": (
                "Five prespecified source-transfer models; no OpenI tuning; all outcomes retained"
            ),
        }
        (output / "result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        status["state"] = "complete"
        save_status()
        return report
    except Exception as error:
        status["state"] = "failed_preserved"
        status["error"] = f"{type(error).__name__}: {error}"
        save_status()
        raise


def main() -> None:
    """Run the fixed secondary comparison after both prerequisites complete."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("reports/metrics/openi_source_head_v1"))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    root = args.root.resolve()
    print(json.dumps(evaluate(root, root / args.output, args.device), indent=2), flush=True)


if __name__ == "__main__":
    main()
