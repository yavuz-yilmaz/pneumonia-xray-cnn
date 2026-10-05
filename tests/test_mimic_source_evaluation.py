"""Test paired inference reuse and preservation of the externally learned source head."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from src.data.download_nih import file_sha256
from src.evaluation import evaluate_mimic_source_head as secondary


def test_identical_models_have_zero_paired_differences() -> None:
    """Every comparison shares identical sampled patient cases."""
    labels = np.array([0, 0, 0, 1, 1, 1])
    scores = {
        name: np.array([0.1, 0.2, 0.6, 0.3, 0.7, 0.9])
        for name in secondary.MODEL_NAMES | {"source_head"}
    }
    result = secondary.paired_differences(labels, scores, dict.fromkeys(scores, 0.5), 25)
    for comparator in result["source_head_minus_comparator"].values():
        for metric in comparator.values():
            assert metric["interval_95"] == [0.0, 0.0]


def test_undefined_precision_is_not_invented() -> None:
    """Do not fill undefined source-minus-baseline precision with zero evidence."""
    scores = dict.fromkeys(secondary.MODEL_NAMES | {"source_head"}, np.zeros(4))
    result = secondary.paired_differences(
        np.array([0, 0, 1, 1]), scores, dict.fromkeys(scores, 0.5), 10
    )
    for comparator in result["source_head_minus_comparator"].values():
        assert comparator["precision"] == {"valid_repetitions": 0, "interval_95": None}


def test_existing_secondary_run_is_preserved(tmp_path: Path) -> None:
    """Existing scored/failed evaluations cannot be silently restarted."""
    with pytest.raises(FileExistsError):
        secondary.evaluate(tmp_path, tmp_path)


def source_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, dict]:
    """Create tiny source tensors with explicit calibration metadata."""
    run = tmp_path / "reports/metrics/nih_mimic_source_head_run_v1"
    run.mkdir(parents=True)
    (run / "status.json").write_text(json.dumps({"state": "complete_source_validation_only"}))
    (run / "registration.json").write_text(json.dumps({"source_and_cohort_sha256": {}}))
    head = {"weight": torch.arange(4, dtype=torch.float32), "bias": torch.tensor(0.3)}
    monkeypatch.setattr(secondary, "load_source_head", lambda path: head)
    features = tmp_path / "models/pretrained/xrv_mimic_ch/features_safe.pt"
    features.parent.mkdir(parents=True)
    torch.save({"feature_state_dict": {"a": torch.tensor([1.0, 2.0])}}, features)
    monkeypatch.setattr(secondary, "MIMIC_FEATURE_SHA256", file_sha256(features))
    protocol = {
        "head_training_on_nih": "none",
        "test_access": "none",
        "openi_access": "none",
        "minimum_recall": 0.9,
    }
    validation = {"threshold": 0.4, "recall": 0.9}
    state = {
        "features.a": torch.tensor([1.0, 2.0]),
        "classifier.weight": torch.stack([torch.zeros(4), head["weight"]]),
        "classifier.bias": torch.tensor([0.0, 0.3]),
    }
    checkpoint = {
        "model_name": "xrv_densenet121",
        "label_mapping": secondary.NIH_LABEL_TO_ID,
        "validation_metrics": validation,
        "decision_threshold": 0.4,
        "training_protocol": protocol,
        "model_state_dict": state,
        "image_size": 224,
        "preprocessing": "resize",
        "normalization": {"mean": (0.5,) * 3, "std": (1 / 2048,) * 3},
    }
    directory = tmp_path / "models/nih_mimic_v1/source_head"
    directory.mkdir(parents=True)
    (directory / "protocol.json").write_text(json.dumps(protocol))
    path = directory / "best_model.pt"
    torch.save(checkpoint, path)
    (directory / "result.json").write_text(
        json.dumps(
            {"status": "complete", "checkpoint_sha256": file_sha256(path), "validation": validation}
        )
    )
    return path, checkpoint


@pytest.mark.parametrize("violation", [None, "features", "classifier", "threshold"])
def test_exact_source_state_is_required(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, violation: str | None
) -> None:
    """Re-signed exports still cannot change the original features or disease head."""
    path, checkpoint = source_fixture(tmp_path, monkeypatch)
    if violation == "features":
        checkpoint["model_state_dict"]["features.a"][0] = 5
    elif violation == "classifier":
        checkpoint["model_state_dict"]["classifier.weight"][1, 0] = 10
    elif violation == "threshold":
        checkpoint["decision_threshold"] = 0.8
    torch.save(checkpoint, path)
    result_path = path.with_name("result.json")
    result = json.loads(result_path.read_bytes())
    result["checkpoint_sha256"] = file_sha256(path)
    result_path.write_text(json.dumps(result))
    if violation is None:
        loaded, digest = secondary.read_source_checkpoint(tmp_path)
        assert loaded["decision_threshold"] == 0.4 and digest == file_sha256(path)
    else:
        with pytest.raises(ValueError):
            secondary.read_source_checkpoint(tmp_path)


def test_secondary_infers_once_and_reuses_four_primary_predictions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run a miniature complete comparison without rescoring existing models."""
    primary = tmp_path / "reports/metrics/openi_external_v1"
    primary.mkdir(parents=True)
    (primary / "status.json").write_text(json.dumps({"state": "complete"}))
    cohort = tmp_path / "data/processed/openi_external_v1"
    cohort.mkdir(parents=True)
    image = tmp_path / "toy.png"
    image.write_bytes(b"toy data; no external patient image")
    labels = np.array([0, 0, 1, 1])
    ids = ["a", "b", "c", "d"]
    with (cohort / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["filepath", "sha256", "case_id", "label_id"])
        writer.writeheader()
        for case, label in zip(ids, labels, strict=True):
            writer.writerow(
                {
                    "filepath": image.as_posix(),
                    "sha256": file_sha256(image),
                    "case_id": case,
                    "label_id": label,
                }
            )
    (cohort / "cohort_audit.json").write_text(
        json.dumps(
            {
                "artifact_sha256": {"manifest.csv": file_sha256(cohort / "manifest.csv")},
                "source_sha256": {},
                "input_sha256": {},
                "target": "toy report code",
                "unit": "case",
            }
        )
    )
    saved_scores = np.array([0.1, 0.8, 0.4, 0.9])
    original_metrics = {}
    for name in secondary.MODEL_NAMES:
        np.savez(
            primary / f"{name}_predictions.npz",
            case_ids=np.array(ids),
            labels=labels,
            probabilities=saved_scores,
        )
        original_metrics[name] = secondary.case_metrics(labels, saved_scores, 0.5)
    (primary / "registration.json").write_text(
        json.dumps(
            {
                "cohort_audit_sha256": file_sha256(cohort / "cohort_audit.json"),
                "thresholds": dict.fromkeys(secondary.MODEL_NAMES, 0.5),
            }
        )
    )
    (primary / "result.json").write_text(json.dumps({"models": original_metrics}))
    source_dir = tmp_path / "models/nih_mimic_v1/source_head"
    source_dir.mkdir(parents=True)
    (source_dir / "result.json").write_text("{}")
    monkeypatch.setattr(secondary, "SOURCE_FILES", ())
    for relative in [
        "src/evaluation/evaluate_mimic_source_head.py",
        "src/training/export_mimic_head.py",
        "docs/nih_mimic_source_head_protocol.md",
    ]:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"pinned fixture source")
    monkeypatch.setattr(
        secondary,
        "read_source_checkpoint",
        lambda root: ({"decision_threshold": 0.5}, "fixed hash"),
    )
    calls = []

    def inference(*args: object) -> tuple:
        calls.append(args)
        return labels, np.array([0.1, 0.2, 0.8, 0.9])

    monkeypatch.setattr(secondary, "predict_images", inference)
    bootstrap = secondary.paired_differences
    monkeypatch.setattr(
        secondary, "paired_differences", lambda *args: bootstrap(*args, repetitions=10)
    )
    report = secondary.evaluate(tmp_path, tmp_path / "evaluation", "cpu")
    assert len(calls) == 1
    assert report["models"]["source_head"]["f1"] == 1
    assert all(report["models"][name] == original_metrics[name] for name in secondary.MODEL_NAMES)
    assert report["primary_candidates_not_replaced"] is True
