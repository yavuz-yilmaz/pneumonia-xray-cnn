"""Check training-only scaling, numerical export and test-free probe execution."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from src.data.dataset import LABEL_TO_ID
from src.training import train_ssmu_probe as module


def test_scaler_is_fit_on_training_only_and_folded_logits_agree() -> None:
    features = np.array([[0, 0], [1, 0], [2, 1], [3, 1]], dtype=np.float32)
    validation = np.array([[100, 10], [-100, -10]], dtype=np.float32)
    head, scores, details = module.fit_head(features, np.array([0, 0, 1, 1]), validation, 0.1)
    assert details["scaler_mean"] == [1.5, 0.5]
    assert details["folded_prediction_max_absolute_error"] < 5e-5
    assert torch.isfinite(head["weight"]).all()
    assert np.isfinite(scores).all()


def test_nonfinite_features_fail_before_fitting() -> None:
    with pytest.raises(ValueError, match="Invalid development"):
        module.fit_head(np.array([[0], [np.nan]]), np.array([0, 1]), np.array([[1]]), 0.1)


class TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.classifier = torch.nn.Linear(2, 2)
        self.pretraining_metadata = {"test_fixture": True}


def test_end_to_end_preserves_infeasible_floors_without_any_test_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifests = tmp_path / "cohort"
    manifests.mkdir()
    hashes = {}
    for split in ("train", "val"):
        path = manifests / f"{split}_manifest.csv"
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["filepath", "label_id", "sha256"])
            writer.writeheader()
            for i, label in enumerate([0, 1, 0, 1]):
                image = tmp_path / f"{split}{i}.png"
                image.write_bytes(bytes([label, i]))
                writer.writerow(
                    {
                        "filepath": str(image),
                        "label_id": label,
                        "sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
                    }
                )
        hashes[split] = hashlib.sha256(path.read_bytes()).hexdigest()
    audit = {
        "task": "ssmu_source_pneumonia",
        "class_to_idx": LABEL_TO_ID,
        "cross_split_near_duplicates": 0,
        "overlaps": {},
        "manifest_sha256": hashes,
    }
    (manifests / "split_audit.json").write_text(json.dumps(audit), encoding="utf-8")
    original, current, feature = (
        tmp_path / name for name in ("original.pt", "current.pt", "features.pt")
    )
    for p in (original, current, feature):
        p.write_bytes(p.name.encode())
    monkeypatch.setattr(
        module,
        "BASELINE_HASHES",
        {
            "original": hashlib.sha256(original.read_bytes()).hexdigest(),
            "current": hashlib.sha256(current.read_bytes()).hexdigest(),
        },
    )
    calls = []
    targets = np.array([0, 1, 0, 1])

    def collect(checkpoint: Path, manifest: Path) -> tuple:
        assert manifest.name == "val_manifest.csv"
        calls.append(manifest.name)
        return targets, np.array([0.1, 0.9, 0.2, 0.8])

    def extract(model: TinyModel, manifest: Path, device: torch.device, **kwargs: bool) -> tuple:
        assert manifest.name in {"train_manifest.csv", "val_manifest.csv"}
        calls.append(manifest.name)
        return np.array([[0, 1], [1, 0], [0, 1], [1, 0]], dtype=np.float32), targets

    monkeypatch.setattr(module, "load_model", lambda p: SimpleNamespace(decision_threshold=0.5))
    monkeypatch.setattr(module, "collect_predictions", collect)
    monkeypatch.setattr(module, "extract", extract)
    monkeypatch.setattr(module, "load_pretrained_features", lambda *args: TinyModel())
    output = tmp_path / "model"
    result = module.train(
        argparse.Namespace(
            manifests=manifests,
            output=output,
            original=original,
            current=current,
            feature_checkpoint=feature,
        )
    )
    assert calls == [
        "val_manifest.csv",
        "val_manifest.csv",
        "train_manifest.csv",
        "val_manifest.csv",
    ]
    assert not (manifests / "test_manifest.csv").exists()
    assert result["joint_threshold"] is None
    assert result["joint_validation"] is None
    assert result["selected_candidate"]["C"] == 0.001
    assert result["test_access"] == "none"
    assert result["full_goal_achieved"] is False
    assert json.loads((output / "status.json").read_bytes())["state"] == "complete_development_only"
    assert len(list(output.glob("c*/best_model.pt"))) == 4
