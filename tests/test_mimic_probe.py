"""Check independent pretraining guards and exported linear-head behavior."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from src.data.dataset import LABEL_TO_ID, ChestXRayDataset
from src.data.nih_dataset import NIH_LABEL_TO_ID, NIHReportDataset
from src.training import train_xray_probe as probe
from torch import nn


def test_nih_refuses_mixed_weights_before_opening_manifests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A contaminated backbone must be refused even if the data audit would pass."""
    monkeypatch.setattr(probe, "verify_manifests", lambda _: pytest.fail("Opened manifests"))
    args = argparse.Namespace(output=tmp_path / "output", task="nih_report_pneumonia")
    with pytest.raises(ValueError, match="mixed NIH/OpenI"):
        probe.train(args)


def test_mimic_refuses_changed_checkpoint_before_deserializing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Filename/source labels cannot override the pinned artifact digest."""
    checkpoint = tmp_path / "features.pt"
    checkpoint.write_bytes(b"changed tensors")
    monkeypatch.setattr(probe.torch, "load", lambda *a, **k: pytest.fail("Deserialized mismatch"))
    with pytest.raises(ValueError, match="reviewed tensor artifact"):
        probe.load_pretrained_features("mimic_ch", checkpoint)


def test_mimic_refuses_wrong_source_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Check both the artifact digest and the claimed original training source."""
    checkpoint = tmp_path / "features.pt"
    torch.save({"pretraining_metadata": {"weights": "densenet121-res224-all"}}, checkpoint)
    monkeypatch.setattr(
        probe, "MIMIC_FEATURE_SHA256", hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    )
    with pytest.raises(ValueError, match="approved MIMIC source"):
        probe.load_pretrained_features("mimic_ch", checkpoint)


@pytest.mark.parametrize("checkpoint", [None, Path("missing.pt")])
def test_mixed_weights_refuse_replacement(checkpoint: Path | None) -> None:
    """Reject missing MIMIC provenance and mixed-source substitutions."""
    if checkpoint is None:
        with pytest.raises(ValueError, match="verified feature checkpoint"):
            probe.load_pretrained_features("mimic_ch", checkpoint)
    else:
        with pytest.raises(ValueError, match="replacement feature checkpoint"):
            probe.load_pretrained_features("all", checkpoint)


@pytest.mark.parametrize("nih_task", [False, True])
def test_exported_heads_match_train_only_scaling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, nih_task: bool
) -> None:
    """Shift validation features to catch scaler leakage and incorrect head folding."""
    generator = np.random.default_rng(17)
    train_features = generator.normal(size=(80, 3)).astype(np.float32)
    train_features[:, 0] *= 7
    train_features[:, 1] += 13
    train_labels = (train_features[:, 0] + train_features[:, 2] > 0).astype(int)
    val_features = generator.normal(size=(20, 3)).astype(np.float32)
    val_features[:, 0] = val_features[:, 0] * 4 + 3
    val_features[:, 1] += 30
    val_labels = (val_features[:, 0] + val_features[:, 2] > 3).astype(int)

    class ToyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.features = nn.Linear(3, 3)
            self.classifier = nn.Linear(3, 2)

    model = ToyModel()
    loader_calls = []
    monkeypatch.setattr(
        probe,
        "load_pretrained_features",
        lambda source, path: loader_calls.append((source, path)) or model,
    )
    mapping = NIH_LABEL_TO_ID if nih_task else LABEL_TO_ID
    monkeypatch.setattr(
        probe,
        "verify_manifests",
        lambda _: {
            "class_to_idx": mapping,
            "fresh_test_exposed_patient_overlap": 0,
            "historical_patient_roles_preserved": True,
        },
    )
    opened = []

    def extract(
        _model: nn.Module,
        manifest: Path,
        _device: torch.device,
        dataset_class: type[ChestXRayDataset],
        cache_images: bool,
    ) -> tuple:
        opened.append((manifest.name, dataset_class))
        assert cache_images is False
        if manifest.name == "train_manifest.csv":
            return train_features, train_labels
        if manifest.name == "val_manifest.csv":
            return val_features, val_labels
        pytest.fail("Opened a holdout manifest")

    monkeypatch.setattr(probe, "extract", extract)
    args = argparse.Namespace(manifests=tmp_path, output=tmp_path / "run", device="cpu")
    if nih_task:
        args.task = "nih_report_pneumonia"
        args.pretraining = "mimic_ch"
    result = probe.train(args)
    assert result["status"] == "complete"
    expected_source = "mimic_ch" if nih_task else "all"
    assert loader_calls == [(expected_source, None)]
    expected_dataset = NIHReportDataset if nih_task else ChestXRayDataset
    assert opened == [
        ("train_manifest.csv", expected_dataset),
        ("val_manifest.csv", expected_dataset),
    ]
    assert all(not p.requires_grad for p in model.features.parameters())
    scaler = StandardScaler().fit(train_features)
    expected_cs = [0.001, 0.01, 0.1, 1.0] if nih_task else [0.01, 0.1, 1.0, 10.0]
    assert [candidate["C"] for candidate in result["candidates"]] == expected_cs
    for candidate in result["candidates"]:
        directory = Path(candidate["directory"])
        saved = torch.load(directory / "best_model.pt", weights_only=True)
        protocol = json.loads((directory / "protocol.json").read_text())
        assert saved["label_mapping"] == mapping
        assert protocol["minimum_recall"] == (0.9 if nih_task else 0.99)
        assert protocol["test_access"] == "none"
        classifier = LogisticRegression(C=candidate["C"], solver="lbfgs", max_iter=2000)
        classifier.fit(scaler.transform(train_features), train_labels)
        state = saved["model_state_dict"]
        logits = (
            torch.from_numpy(val_features) @ state["classifier.weight"].T + state["classifier.bias"]
        )
        assert np.allclose(
            logits.softmax(1)[:, 1].numpy(),
            classifier.predict_proba(scaler.transform(val_features))[:, 1],
            rtol=1e-5,
            atol=1e-6,
        )
    if nih_task:
        expected = max(
            result["candidates"], key=lambda c: (c["validation"]["average_precision"], -c["C"])
        )
        assert result["selected_candidate"] == expected
