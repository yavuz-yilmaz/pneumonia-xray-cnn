"""Keep report-pneumonia semantics distinct from the original NORMAL/PNEUMONIA task."""

import csv
from pathlib import Path

import pytest
from PIL import Image
from src.data.dataset import ChestXRayDataset, XRayDatasetError
from src.data.nih_dataset import NIHReportDataset


def manifest(tmp_path: Path, label: str, label_id: int) -> Path:
    image = tmp_path / "00000001_000.png"
    Image.new("L", (8, 8), color=128).save(image)
    path = tmp_path / "train_manifest.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["filepath", "split", "label", "label_id"])
        writer.writeheader()
        writer.writerow(
            {"filepath": image.as_posix(), "split": "train", "label": label, "label_id": label_id}
        )
    return path


def test_nih_negative_keeps_explicit_semantics(tmp_path: Path) -> None:
    dataset = NIHReportDataset(manifest(tmp_path, "NO_REPORTED_PNEUMONIA", 0))
    assert dataset.get_sample(0).label == "NO_REPORTED_PNEUMONIA"
    assert dataset.get_sample(0).label_id == 0


def test_original_loader_rejects_nih_task_labels(tmp_path: Path) -> None:
    with pytest.raises(XRayDatasetError, match="label"):
        ChestXRayDataset(manifest(tmp_path, "NO_REPORTED_PNEUMONIA", 0))


def test_nih_loader_rejects_normal_as_its_negative_class(tmp_path: Path) -> None:
    with pytest.raises(XRayDatasetError, match="label"):
        NIHReportDataset(manifest(tmp_path, "NORMAL", 0))


def test_nih_mapping_conflict_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(XRayDatasetError, match="mapping"):
        NIHReportDataset(manifest(tmp_path, "REPORT_PNEUMONIA", 0))
