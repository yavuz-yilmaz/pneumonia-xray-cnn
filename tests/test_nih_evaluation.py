"""Test prospective selection isolation and aligned patient-level comparisons."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from src.data.dataset import LABEL_TO_ID
from src.data.nih_dataset import NIH_LABEL_TO_ID
from src.evaluation import evaluate_nih


def audit() -> dict:
    return {
        "class_to_idx": NIH_LABEL_TO_ID,
        "historical_patient_roles_preserved": True,
        "fresh_test_exposed_patient_overlap": 0,
        "cross_split_near_duplicates": 0,
        "overlaps": {"train/test": {"connected_components": 0}},
        "manifest_sha256": {"train": "train-hash", "val": "val-hash", "test": "test-hash"},
    }


def test_equal_models_have_zero_paired_difference_intervals() -> None:
    labels = np.array([0, 0, 1, 1, 0, 1])
    scores = np.array([0.1, 0.7, 0.9, 0.6, 0.2, 0.8])
    predictions = {name: scores.copy() for name in ("selected", "original", "current")}
    thresholds = dict.fromkeys(predictions, 0.5)
    result = evaluate_nih.paired_cluster_intervals(
        labels, predictions, thresholds, np.array(["a", "a", "b", "b", "c", "c"]), 40
    )
    assert result["valid_repetitions"] > 0
    for intervals in result["selected_minus_baseline_intervals_95"].values():
        assert all(bounds == [0.0, 0.0] for bounds in intervals.values())
    assert "average_precision" in result["model_intervals_95"]["selected"]


def test_unaligned_predictions_are_rejected() -> None:
    scores = {name: np.array([0.1, 0.9]) for name in ("selected", "original", "current")}
    scores["current"] = np.array([0.1])
    with pytest.raises(ValueError, match="not aligned"):
        evaluate_nih.paired_cluster_intervals(
            np.array([0, 1]), scores, dict.fromkeys(scores, 0.5), np.array(["a", "b"]), 10
        )


def test_selection_uses_validation_only_and_locks_baselines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    experiments = []
    for model in ("resnet18", "efficientnet_v2_s"):
        directory = tmp_path / model
        directory.mkdir()
        protocol = {
            "task": "nih_report_pneumonia",
            "minimum_recall": 0.9,
            "feature_checkpoint": None,
            "test_access": "none",
            "model": model,
            "data_audit": audit(),
        }
        (directory / "protocol.json").write_text(json.dumps(protocol))
        (directory / "result.json").write_text(json.dumps({"status": "complete"}))
        torch.save(
            {
                "model_name": model,
                "label_mapping": NIH_LABEL_TO_ID,
                "image_size": 32,
                "model_state_dict": {},
                "training_protocol": protocol,
            },
            directory / "best_model.pt",
        )
        experiments.append(directory)
    baselines = {}
    for name, threshold in (("original", 0.7), ("current", 0.882)):
        path = tmp_path / f"{name}.pt"
        torch.save(
            {
                "model_name": "resnet18",
                "label_mapping": LABEL_TO_ID,
                "image_size": 32,
                "decision_threshold": threshold,
            },
            path,
        )
        baselines[name] = path
    accessed = []

    def predictions(path: Path, manifest: Path) -> tuple:
        accessed.append(manifest.name)
        assert manifest.name == "val_manifest.csv"
        scores = [0.8, 0.7, 0.6, 0.1] if "resnet18" in path.as_posix() else [0.95, 0.05, 0.8, 0.1]
        return np.array([1, 0, 1, 0]), np.array(scores), 0.2

    monkeypatch.setattr(evaluate_nih, "verify_manifests", lambda _: audit())
    monkeypatch.setattr(evaluate_nih, "collect_predictions", predictions)
    selection = evaluate_nih.select_model(
        experiments, tmp_path / "no_test_file_exists", tmp_path / "selected", baselines
    )
    assert accessed == ["val_manifest.csv", "val_manifest.csv"]
    assert "efficientnet_v2_s" in selection["winner"]["checkpoint"]
    assert selection["test_used_for_selection"] is False
    assert selection["baselines"]["original"]["threshold"] == 0.7
    assert selection["baselines"]["current"]["threshold"] == 0.882


def test_existing_test_lock_prevents_repeat_scoring(tmp_path: Path) -> None:
    selected = tmp_path / "selected"
    selected.mkdir()
    (selected / "test_started.json").write_text("{}")
    with pytest.raises(FileExistsError, match="already started"):
        evaluate_nih.evaluate_selected(selected, tmp_path / "not_read", tmp_path / "output")


def test_nih_source_pretrained_checkpoint_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "xrv.pt"
    torch.save({"model_name": "xrv_densenet121", "label_mapping": LABEL_TO_ID}, path)
    with pytest.raises(ValueError, match="NIH-pretrained"):
        evaluate_nih.read_checkpoint(path)
