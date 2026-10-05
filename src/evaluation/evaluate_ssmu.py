"""Freeze the registered SSMU classifier, then compare its reserved test once."""

from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from src.data.dataset import LABEL_TO_ID
from src.data.download_nih import file_sha256
from src.evaluation.decision import binary_metrics
from src.evaluation.evaluate_clean import collect_predictions
from src.evaluation.evaluate_nih import paired_cluster_intervals
from src.inference.predict import load_model
from src.training.calibrate_joint_metrics import select_joint_threshold
from src.training.train_ssmu_probe import BASELINE_HASHES
from src.training.train_xray_probe import MIMIC_FEATURE_SHA256


def freeze_selection(run: Path, manifests: Path, selected: Path) -> dict:
    """Review saved development predictions without any test manifest access."""
    if selected.exists():
        raise FileExistsError("Selection already frozen; inspect saved state")
    result = json.loads((run / "result.json").read_bytes())
    if result["status"] != "complete_development_only" or result["joint_threshold"] is None:
        raise ValueError("Completed feasible validation policy required before test access")
    protocol = json.loads((run / "protocol.json").read_bytes())
    for name, digest in protocol["input_sha256"].items():
        if file_sha256(Path(name)) != digest:
            raise ValueError("Registered training input changed")
    audit = json.loads((manifests / "split_audit.json").read_bytes())
    if (
        audit["task"] != "ssmu_source_pneumonia"
        or audit["class_to_idx"] != LABEL_TO_ID
        or audit["cross_split_near_duplicates"]
        or any(any(v.values()) for v in audit["overlaps"].values())
    ):
        raise ValueError("SSMU group/duplicate separation audit failed")
    candidates = [json.loads(p.read_bytes()) for p in sorted(run.glob("c*/result.json"))]
    if {c["C"] for c in candidates} != set(protocol["regularizations"]) or len(candidates) != 4:
        raise ValueError("Incomplete registered candidate set")
    for candidate in candidates:
        prediction = Path(candidate["directory"]) / "validation_predictions.npz"
        with np.load(prediction, allow_pickle=False) as saved:
            recalculated = binary_metrics(
                saved["labels"], saved["probabilities"], candidate["validation"]["threshold"]
            )
        if recalculated != candidate["validation"]:
            raise ValueError("Candidate metrics differ from saved predictions")
    winner = max(candidates, key=lambda c: (c["validation"]["average_precision"], -c["C"]))
    if winner != result["selected_candidate"]:
        raise ValueError("Saved AP selection differs")
    path = Path(winner["directory"]) / "best_model.pt"
    if file_sha256(path) != result["selected_checkpoint_sha256"]:
        raise ValueError("Selected source checkpoint changed")
    with (manifests / "val_manifest.csv").open(newline="", encoding="utf-8") as stream:
        expected = np.asarray([int(r["label_id"]) for r in csv.DictReader(stream)])
    with np.load(path.with_name("validation_predictions.npz"), allow_pickle=False) as saved:
        labels, scores = saved["labels"], saved["probabilities"]
    if not np.array_equal(labels, expected):
        raise ValueError("Saved validation label order differs")
    for name, metric in result["baselines"].items():
        with np.load(run / f"{name}_validation_predictions.npz", allow_pickle=False) as saved:
            if not np.array_equal(saved["labels"], expected):
                raise ValueError("Saved baseline label order differs")
            recalculated = binary_metrics(expected, saved["probabilities"], metric["threshold"])
        if recalculated != metric:
            raise ValueError("Baseline metrics differ from saved probabilities")
    threshold = select_joint_threshold(labels, scores, result["baselines"])
    if (
        threshold != result["joint_threshold"]
        or binary_metrics(labels, scores, threshold) != result["joint_validation"]
    ):
        raise ValueError("Registered joint validation policy differs")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    feature = Path("models/pretrained/xrv_mimic_ch/features_safe.pt")
    if file_sha256(feature) != MIMIC_FEATURE_SHA256:
        raise ValueError("MIMIC feature artifact changed")
    original_features = torch.load(feature, map_location="cpu", weights_only=True)[
        "feature_state_dict"
    ]
    tensors = checkpoint["model_state_dict"]
    if not all(torch.isfinite(t).all() for t in tensors.values()):
        raise ValueError("Nonfinite selected tensors")
    for name, tensor in original_features.items():
        if not torch.equal(tensor, tensors[f"features.{name}"]):
            raise ValueError("Frozen MIMIC feature tensor changed")
    with np.load(run / "features.npz", allow_pickle=False) as features:
        if not np.array_equal(features["val_labels"], expected):
            raise ValueError("Validation feature order differs")
        logits = torch.nn.functional.linear(
            torch.from_numpy(features["val"]).float(),
            tensors["classifier.weight"],
            tensors["classifier.bias"],
        )
        readback_scores = logits.softmax(1)[:, 1].numpy()
    if not np.allclose(scores, readback_scores, atol=5e-6, rtol=1e-5):
        raise ValueError("Selected classifier differs from saved feature predictions")
    checkpoint["decision_threshold"] = threshold
    checkpoint["validation_metrics"] = result["joint_validation"]
    checkpoint["model_version"] += "_joint"
    checkpoint["operating_policy"] = {
        "selection": "joint validation floors",
        "threshold_fit": "validation only",
    }
    selected.mkdir(parents=True)
    torch.save(checkpoint, selected / "best_model.pt")
    summary = {
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_C": winner["C"],
        "source_checkpoint_sha256": file_sha256(path),
        "selected_checkpoint_sha256": file_sha256(selected / "best_model.pt"),
        "threshold": threshold,
        "validation": result["joint_validation"],
        "data_audit": audit,
        "patient_identity_verified": False,
        "clinical_labels_verified": False,
        "test_access": "none",
    }
    (selected / "selection.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def evaluate(run: Path, manifests: Path, output: Path) -> dict:
    """Register checkpoint/cohort hashes before one-time full-precision inference."""
    if output.exists() or (run / "test_started.json").exists():
        raise FileExistsError("SSMU test already started; never silently rescore")
    selected = run / "selected_joint"
    selection = freeze_selection(run, manifests, selected)
    paths = {
        "selected": selected / "best_model.pt",
        "original": Path("models/best_model.pt"),
        "current": Path("models/clean_v3/selected_matched98/best_model.pt"),
    }
    hashes = {name: file_sha256(path) for name, path in paths.items()}
    if hashes["selected"] != selection["selected_checkpoint_sha256"] or any(
        hashes[name] != BASELINE_HASHES[name] for name in BASELINE_HASHES
    ):
        raise ValueError("Frozen checkpoint changed")
    manifest = manifests / "test_manifest.csv"
    if file_sha256(manifest) != selection["data_audit"]["manifest_sha256"]["test"]:
        raise ValueError("Reserved test manifest changed")
    sources = {
        str(p): file_sha256(p)
        for p in (
            Path(__file__),
            Path("docs/ssmu_test_protocol.md"),
            Path("src/evaluation/evaluate_clean.py"),
            Path("src/evaluation/evaluate_nih.py"),
            Path("src/evaluation/decision.py"),
            Path("src/inference/predict.py"),
        )
    }
    registration = {
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "process_id": os.getpid(),
        "checkpoint_sha256": hashes,
        "test_manifest_sha256": file_sha256(manifest),
        "evaluator_source_sha256": sources,
        "selected_threshold": selection["threshold"],
        "bootstrap_repetitions": 1000,
        "patient_identity_verified": False,
        "clinical_labels_verified": False,
    }
    output.mkdir(parents=True)
    (output / "registration.json").write_text(json.dumps(registration, indent=2), encoding="utf-8")
    (run / "test_started.json").open("x", encoding="utf-8").write(
        json.dumps(registration, indent=2)
    )
    status = {"process_id": os.getpid(), "state": "scoring_once"}
    (output / "status.json").write_text(json.dumps(status), encoding="utf-8")
    try:
        with manifest.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        labels = np.asarray([int(r["label_id"]) for r in rows])
        groups = np.asarray([r["group_id"] for r in rows])
        scores, thresholds, metrics = {}, {}, {}
        torch.set_num_threads(4)
        for name, path in paths.items():
            loaded = load_model(path)
            thresholds[name] = loaded.decision_threshold
            del loaded
            targets, probabilities = collect_predictions(path, manifest)
            if not np.array_equal(targets, labels):
                raise ValueError("Test prediction label order differs")
            scores[name] = probabilities
            metrics[name] = binary_metrics(labels, probabilities, thresholds[name])
            np.savez(
                output / f"{name}_predictions.npz",
                labels=labels,
                probabilities=probabilities,
                groups=groups,
                source_members=np.asarray([r["source_member"] for r in rows]),
            )
            print(json.dumps({"model": name, "test": metrics[name]}), flush=True)
        intervals = paired_cluster_intervals(labels, scores, thresholds, groups, repetitions=1000)
        intervals["grouping"] = (
            "Global filename-number proxies joined by documented duplicates; "
            "patient identities unverified"
        )
        report = {
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "complete_one_time_source_test",
            "images": len(labels),
            "positive_images": int(labels.sum()),
            "proxy_groups": len(np.unique(groups)),
            "metrics": metrics,
            "paired_intervals": intervals,
            "checkpoint_sha256": hashes,
            "patient_identity_verified": False,
            "clinical_labels_verified": False,
            "full_goal_achieved": False,
            "deployment_changed": False,
            "test_manifest_sha256": file_sha256(manifest),
        }
        if any(file_sha256(Path(name)) != digest for name, digest in sources.items()):
            raise ValueError("Evaluation source drifted during inference")
        (output / "test_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        status["state"] = "complete_one_time_source_test"
        return report
    except Exception as error:
        status.update(state="failed_preserved", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        (output / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("models/ssmu_probe_v1"))
    parser.add_argument("--manifests", type=Path, default=Path("data/processed/ssmu_frontal_v1"))
    parser.add_argument("--output", type=Path, default=Path("reports/metrics/ssmu_probe_v1"))
    result = evaluate(**vars(parser.parse_args()))
    print(
        json.dumps({k: v for k, v in result.items() if k != "paired_intervals"}, indent=2),
        flush=True,
    )


if __name__ == "__main__":
    main()
