"""Regression checks for patient grouping and image leakage prevention."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from src.data.clean_split import (
    Components,
    assign_splits,
    audit_splits,
    connect_records,
    fingerprint,
    group_id,
)
from src.evaluation.decision import select_threshold
from src.training.train_clean import verify_manifests


def test_patient_grouping_is_conservative_across_subtypes() -> None:
    assert group_id(Path("person100_bacteria_2.jpeg")) == group_id(Path("person100_virus_9.jpeg"))
    assert group_id(Path("NORMAL2-IM-1234-0001.jpeg")) == group_id(
        Path("NORMAL2-IM-1234-0002.jpeg")
    )
    assert group_id(Path("IM-1234-0001.jpeg")) != group_id(Path("NORMAL2-IM-1234-0001.jpeg"))
    with pytest.raises(ValueError, match="Unknown patient"):
        group_id(Path("unknown.png"))


def test_decoded_duplicates_are_joined_even_when_encoding_differs(tmp_path: Path) -> None:
    directory = tmp_path / "train" / "NORMAL"
    directory.mkdir(parents=True)
    pixels = np.random.default_rng(5).integers(0, 256, (64, 64), dtype=np.uint8)
    first, second = directory / "IM-0001-0001.png", directory / "IM-0002-0001.png"
    Image.fromarray(pixels).save(first, compress_level=0)
    Image.fromarray(pixels).save(second, compress_level=9)
    records = [fingerprint(first), fingerprint(second)]
    assert records[0]["sha256"] != records[1]["sha256"]
    components, pairs = connect_records(records)
    assert components.root(0) == components.root(1)
    assert any(pair["kind"] == "pixel_sha256" for pair in pairs)


def test_test_connected_group_is_excluded_without_removing_test() -> None:
    records = [
        {
            "source_split": "train",
            "label_id": i % 2,
            "label": str(i % 2),
            "pixel_sha256": str(i),
            "sha256": str(i),
            "patient_proxy": str(i),
        }
        for i in range(40)
    ]
    records.append({**records[0], "source_split": "test"})
    components = Components(len(records))
    components.join(0, 40)
    splits = assign_splits(records, components, 42)
    assert splits["test"] == [40]
    assert 0 in splits["excluded"]
    assert 0 not in splits["train"] + splits["val"]
    audit_splits(records, splits, components, [])


def test_audit_rejects_cross_split_patient_overlap() -> None:
    records = [
        {"patient_proxy": "same", "sha256": str(i), "pixel_sha256": str(i), "label": "NORMAL"}
        for i in range(2)
    ]
    with pytest.raises(ValueError, match="Split leakage"):
        audit_splits(records, {"train": [0], "val": [1], "test": []}, Components(2), [])


def test_operating_point_meets_recall_constraint_and_maximizes_specificity() -> None:
    labels = np.array([0, 0, 0, 1, 1, 1])
    probabilities = np.array([0.1, 0.2, 0.7, 0.6, 0.8, 0.9])
    threshold = select_threshold(labels, probabilities, 1.0)
    assert threshold == pytest.approx(0.6)
    assert (probabilities[labels == 1] >= threshold).all()


def test_threshold_rejects_nan() -> None:
    with pytest.raises(ValueError, match="finite"):
        select_threshold(np.array([0, 1]), np.array([0.2, np.nan]))


@pytest.mark.parametrize("changed_item", ["manifest", "image"])
def test_training_rejects_changes_after_dataset_audit(tmp_path: Path, changed_item: str) -> None:
    manifest_hashes = {}
    for split in ("train", "val"):
        source = tmp_path / f"{split}.png"
        Image.new("RGB", (8, 8), color="gray").save(source)
        manifest = tmp_path / f"{split}_manifest.csv"
        with manifest.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["filepath", "sha256"])
            writer.writeheader()
            writer.writerow(
                {
                    "filepath": source.as_posix(),
                    "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                }
            )
        manifest_hashes[split] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    audit = {"cross_split_near_duplicates": 0, "overlaps": {}, "manifest_sha256": manifest_hashes}
    (tmp_path / "split_audit.json").write_text(json.dumps(audit), encoding="utf-8")
    verify_manifests(tmp_path)
    if changed_item == "manifest":
        with (tmp_path / "train_manifest.csv").open("a", encoding="utf-8") as stream:
            stream.write("\n")
    else:
        Image.new("RGB", (8, 8), color="white").save(tmp_path / "train.png")
    with pytest.raises(ValueError, match="changed after audit"):
        verify_manifests(tmp_path)
