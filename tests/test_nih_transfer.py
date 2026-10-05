"""Verify selective adaptation and prevention of pretrained source drift."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from src.data.nih_dataset import NIH_LABEL_TO_ID
from src.evaluation.decision import binary_metrics
from src.training import train_nih_transfer as transfer
from src.training.models import XRayDenseNet121
from torch import nn


def test_tail_update_preserves_early_features_and_all_batchnorm_buffers() -> None:
    """Backpropagation must reach the tail while frozen statistics remain unchanged."""
    torch.set_num_threads(2)
    model = XRayDenseNet121(pretrained=False)
    transfer.configure_tail(model)
    before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.01)
    images = torch.randn(2, 3, 64, 64)
    loss = nn.functional.cross_entropy(model(images), torch.tensor([0, 1]))
    loss.backward()
    optimizer.step()
    changed = {
        name for name, tensor in model.state_dict().items() if not torch.equal(tensor, before[name])
    }
    assert any(name.startswith("features.denseblock4.") for name in changed)
    assert any(name.startswith("classifier.") for name in changed)
    assert all(name.startswith(("features.denseblock4.", "classifier.")) for name in changed)
    assert all(
        not name.endswith(("running_mean", "running_var", "num_batches_tracked"))
        for name in changed
    )
    assert all(not p.requires_grad for p in model.features.denseblock3.parameters())
    # Validation switches the whole model to eval; each next epoch must restore
    # trainable layers while keeping every BatchNorm module in eval.
    model.eval()
    transfer.configure_tail(model)
    assert model.features.denseblock4.training
    assert all(
        not module.training
        for module in model.modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm)
    )


def test_changed_head_is_rejected_before_feature_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A filename alone cannot establish provenance."""
    head = tmp_path / "changed.pt"
    head.write_bytes(b"different artifact")
    monkeypatch.setattr(
        transfer,
        "load_pretrained_features",
        lambda *args: pytest.fail("Loaded features before rejecting head"),
    )
    with pytest.raises(ValueError, match="registered artifact"):
        transfer.load_initial_model(tmp_path / "features.pt", head)


def test_adapted_backbone_cannot_be_used_as_unchanged_initial_features(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject a checkpoint with correct task labels but altered pretrained features."""
    head = tmp_path / "head.pt"
    head.write_bytes(b"placeholder")
    monkeypatch.setattr(transfer, "file_sha256", lambda _: transfer.HEAD_SHA256)
    model = XRayDenseNet121(pretrained=False)
    state = {k: v.clone() for k, v in model.state_dict().items()}
    state["features.conv0.weight"] += 1
    monkeypatch.setattr(transfer, "load_pretrained_features", lambda *args: model)
    monkeypatch.setattr(
        transfer.torch,
        "load",
        lambda *args, **kwargs: {
            "label_mapping": NIH_LABEL_TO_ID,
            "model_state_dict": state,
        },
    )
    with pytest.raises(ValueError, match="unchanged MIMIC backbone"):
        transfer.load_initial_model(tmp_path / "features.pt", head)


@pytest.mark.parametrize("nonfinite", [False, True])
def test_complete_development_run_never_opens_holdout_and_retains_infeasible_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, nonfinite: bool
) -> None:
    """Exercise actual manifest checks, gradients, selection and final policy on tiny data."""

    class TinyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.features = nn.ModuleDict(
                {
                    "early": nn.Conv2d(3, 2, 1),
                    "denseblock4": nn.Conv2d(2, 2, 1),
                }
            )
            self.classifier = nn.Linear(2, 2)
            self.pretraining_metadata = {"weights": "densenet121-res224-mimic_ch"}

        def forward(self, images: torch.Tensor) -> torch.Tensor:
            features = self.features["early"](images)
            features = self.features["denseblock4"](features).mean((2, 3))
            logits = self.classifier(features)
            return logits * float("nan") if nonfinite else logits

    manifests = tmp_path / "manifests"
    manifests.mkdir()
    digests = {}
    for split in ("train", "val"):
        records = []
        for label in (0, 1):
            image_path = tmp_path / f"{split}_{label}.png"
            Image.new("RGB", (32, 32), (30 + label * 180,) * 3).save(image_path)
            records.append(
                {
                    "filepath": str(image_path),
                    "split": split,
                    "label": next(k for k, v in NIH_LABEL_TO_ID.items() if v == label),
                    "label_id": label,
                    "sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
                }
            )
        path = manifests / f"{split}_manifest.csv"
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        digests[split] = hashlib.sha256(path.read_bytes()).hexdigest()
    (manifests / "split_audit.json").write_text(
        json.dumps(
            {
                "overlaps": {"train/val": {"patient_proxy": 0}},
                "cross_split_near_duplicates": 0,
                "manifest_sha256": digests,
                "class_to_idx": NIH_LABEL_TO_ID,
                "historical_patient_roles_preserved": True,
                "fresh_test_exposed_patient_overlap": 0,
            }
        ),
        encoding="utf-8",
    )
    labels, scores = np.array([0, 1]), np.array([0.1, 0.9])
    baseline_metrics = binary_metrics(labels, scores, 0.5)
    baselines = tmp_path / "baseline_result.json"
    baselines.write_text(
        json.dumps(
            {
                "validation_manifest_sha256": digests["val"],
                "baselines": dict.fromkeys(("original", "current"), baseline_metrics),
            }
        ),
        encoding="utf-8",
    )
    for name in ("original", "current"):
        np.savez(
            tmp_path / f"{name}_validation_predictions.npz", labels=labels, probabilities=scores
        )
    feature_path, head_path = tmp_path / "features.pt", tmp_path / "head.pt"
    feature_path.write_bytes(b"test fixture")
    head_path.write_bytes(b"test fixture")
    model = TinyModel()
    monkeypatch.setattr(transfer, "load_initial_model", lambda *args: model)
    monkeypatch.setattr(transfer.torch.cuda, "is_available", lambda: False)
    opened = []
    original_open = Path.open

    def guarded_open(path: Path, *args: object, **kwargs: object) -> object:
        assert path.name != "test_manifest.csv", "Opened NIH test"
        assert "openi" not in str(path).lower(), "Opened OpenI data"
        opened.append(path)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    args = argparse.Namespace(
        output=tmp_path / "run",
        manifests=manifests,
        feature_checkpoint=feature_path,
        head_checkpoint=head_path,
        baseline_result=baselines,
        epochs=1,
        epoch_samples=4,
        batch_size=2,
    )
    if nonfinite:
        with pytest.raises(RuntimeError, match="Nonfinite training logits"):
            transfer.train(args)
        state = json.loads((args.output / "status.json").read_bytes())
        assert state["state"] == "failed_preserved"
        assert not (args.output / "best_model.pt").exists()
        return
    result = transfer.train(args)
    assert result["status"] == "complete"
    assert result["joint_threshold"] is None
    assert result["joint_validation"] is None
    assert not result["external_improvement_established"]
    assert (args.output / "best_model.pt").is_file()
    state = json.loads((args.output / "status.json").read_bytes())
    assert state["state"] == "complete_development_only"
    assert state["completed_epochs"] == 1
    assert {"train_manifest.csv", "val_manifest.csv"} <= {p.name for p in opened}
