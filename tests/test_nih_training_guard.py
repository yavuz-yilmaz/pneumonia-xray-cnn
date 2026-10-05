"""Prevent unregistered NIH operating policies and source-pretrained leakage."""

import argparse
from pathlib import Path

import pytest
from src.data.nih_dataset import NIH_LABEL_TO_ID
from src.training import train_clean


def arguments(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        output=tmp_path / "new_run",
        manifests=tmp_path / "not_opened",
        task="nih_report_pneumonia",
        model="resnet18",
        feature_checkpoint=None,
        minimum_recall=0.9,
    )


def audit() -> dict:
    return {
        "class_to_idx": NIH_LABEL_TO_ID,
        "fresh_test_exposed_patient_overlap": 0,
        "historical_patient_roles_preserved": True,
    }


def test_nih_pretrained_xrv_is_rejected_before_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(train_clean, "verify_manifests", lambda _: audit())
    args = arguments(tmp_path)
    args.model = "xrv_densenet121"
    with pytest.raises(ValueError, match="ImageNet-only"):
        train_clean.train(args)
    assert not args.output.exists()


def test_changed_nih_recall_policy_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(train_clean, "verify_manifests", lambda _: audit())
    args = arguments(tmp_path)
    args.minimum_recall = 0.99
    with pytest.raises(ValueError, match="fixes minimum"):
        train_clean.train(args)


def test_missing_historical_role_guarantee_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    incomplete = audit()
    incomplete.pop("historical_patient_roles_preserved")
    monkeypatch.setattr(train_clean, "verify_manifests", lambda _: incomplete)
    with pytest.raises(ValueError, match="preserve previous"):
        train_clean.train(arguments(tmp_path))
