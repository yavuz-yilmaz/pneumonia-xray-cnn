"""Preserve the source classifier and calibrate without training/test feature access."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from src.data.download_nih import file_sha256
from src.training import export_mimic_head as source_head
from torch import nn


def source_metadata() -> dict:
    """Create native metadata for a pinned, correctly ordered pneumonia row."""
    return {
        "weights": "densenet121-res224-mimic_ch",
        "source_sha256": source_head.MIMIC_SOURCE_SHA256,
        "feature_checkpoint_sha256": source_head.MIMIC_FEATURE_SHA256,
        "pathology": "Pneumonia",
        "source_row": 8,
        "publisher_supported_label": "Pneumonia",
        "source_targets": ["other"] * 8 + ["Pneumonia"] + ["other"] * 9,
    }


def test_source_head_hash_checked_before_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject drift before any deserialization."""
    path = tmp_path / "head.pt"
    path.write_bytes(b"altered")
    monkeypatch.setattr(torch, "load", lambda *a, **kw: pytest.fail("untrusted file loaded"))
    with pytest.raises(ValueError, match="reviewed artifact"):
        source_head.load_source_head(path)


@pytest.mark.parametrize("violation", ["pathology", "row", "mixed", "shape", "nonfinite"])
def test_head_provenance_and_parameter_gates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, violation: str
) -> None:
    """A valid tensor file still cannot supply the wrong disease or source."""
    metadata = source_metadata()
    weight = torch.ones(1024)
    if violation == "pathology":
        metadata["pathology"] = "Consolidation"
    elif violation == "row":
        metadata["source_row"] = 7
    elif violation == "mixed":
        metadata["weights"] = "densenet121-res224-all"
    elif violation == "shape":
        weight = torch.ones(18, 1024)
    else:
        weight[0] = float("nan")
    path = tmp_path / "head.pt"
    torch.save({"metadata": metadata, "weight": weight, "bias": torch.tensor(0.1)}, path)
    monkeypatch.setattr(source_head, "HEAD_SHA256", file_sha256(path))
    with pytest.raises(ValueError):
        source_head.load_source_head(path)


def test_existing_output_and_unfinished_prerequisite_refused(tmp_path: Path) -> None:
    """Avoid replacing an experiment or exporting partial development features."""
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(FileExistsError):
        source_head.export(tmp_path, output)
    run = tmp_path / "reports/metrics/nih_mimic_run_v1"
    run.mkdir(parents=True)
    (run / "status.json").write_text(json.dumps({"state": "training"}))
    with pytest.raises(ValueError, match="finish first"):
        source_head.export(tmp_path, tmp_path / "new")
    assert not (tmp_path / "new").exists()


def test_export_preserves_original_head_and_uses_only_validation_features(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing train/test arrays force the exporter to honor its declared scope."""
    run = tmp_path / "reports/metrics/nih_mimic_run_v1"
    run.mkdir(parents=True)
    (run / "status.json").write_text(json.dumps({"state": "complete_train_validation_only"}))
    manifests = tmp_path / "data/processed/nih_pneumonia_v1"
    manifests.mkdir(parents=True)
    manifest = manifests / "val_manifest.csv"
    manifest.write_text("label_id\n0\n1\n0\n1\n")
    (run / "registration.json").write_text(
        json.dumps(
            {
                "source_sha256": {
                    "data/processed/nih_pneumonia_v1/val_manifest.csv": file_sha256(manifest)
                },
            }
        )
    )
    frozen = tmp_path / "models/nih_mimic_v1/frozen"
    frozen.mkdir(parents=True)
    (frozen / "result.json").write_text(json.dumps({"status": "complete"}))
    features = np.zeros((4, 1024), dtype=np.float32)
    features[:, 0] = [-2, 2, -1, 1]
    labels = np.array([0, 1, 0, 1])
    np.savez(frozen / "features.npz", val=features, val_labels=labels)
    directory = tmp_path / "models/pretrained/xrv_mimic_ch"
    directory.mkdir(parents=True)
    head = directory / "pneumonia_head_safe.pt"
    weight = torch.zeros(1024)
    weight[0] = 0.8
    bias = torch.tensor(0.1)
    torch.save({"metadata": source_metadata(), "weight": weight, "bias": bias}, head)
    monkeypatch.setattr(source_head, "HEAD_SHA256", file_sha256(head))

    class ToyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.features = nn.Linear(1, 1)
            self.classifier = nn.Linear(1024, 2)
            self.pretraining_metadata = {"weights": "densenet121-res224-mimic_ch"}

    model = ToyModel()
    monkeypatch.setattr(source_head, "load_pretrained_features", lambda *a: model)
    output = tmp_path / "export"
    summary = source_head.export(tmp_path, output)
    saved = torch.load(output / "best_model.pt", map_location="cpu", weights_only=True)
    np.testing.assert_array_equal(saved["model_state_dict"]["classifier.weight"][1], weight)
    assert saved["model_state_dict"]["classifier.bias"][1] == bias
    assert not saved["model_state_dict"]["classifier.weight"][0].count_nonzero()
    assert not any(p.requires_grad for p in model.parameters())
    with np.load(output / "validation_predictions.npz") as predictions:
        expected = (torch.from_numpy(features) @ weight + bias).sigmoid().numpy()
        np.testing.assert_allclose(predictions["probabilities"], expected, rtol=1e-6)
    assert summary["validation"]["recall"] >= 0.9
    assert summary["not_a_new_nih_fit"] is True
    assert saved["training_protocol"]["head_training_on_nih"] == "none"
    assert summary["test_access"] == summary["openi_access"] == "none"
