import hashlib
import json
from pathlib import Path

import pytest
from src.evaluation.evaluate_rsna import verified_images


def make_cohort(directory: Path) -> Path:
    """Build minimal integrity-linked metadata without running model inference."""
    image = directory / "image.png"
    image.write_bytes(b"image-content")
    cohort = directory / "cohort.json"
    records = [
        {
            "nih_patient_id": "00000001",
            "source_label": "Normal",
            "label_id": 0,
            "filepath": str(image),
            "sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
        }
    ]
    cohort.write_text(json.dumps(records), encoding="utf-8")
    cohort_hash = hashlib.sha256(cohort.read_bytes()).hexdigest()
    (directory / "protocol.json").write_text(json.dumps({"cohort_sha256": cohort_hash}))
    manifest = directory / "images.json"
    manifest.write_text(json.dumps(records))
    audit = {
        "overlaps": [],
        "cohort_sha256": cohort_hash,
        "images": 1,
        "images_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    }
    (directory / "image_audit.json").write_text(json.dumps(audit))
    return image


def test_external_evaluation_rejects_modified_image(tmp_path: Path) -> None:
    image = make_cohort(tmp_path)
    assert len(verified_images(tmp_path)) == 1
    image.write_bytes(b"changed")
    with pytest.raises(ValueError, match="External image changed"):
        verified_images(tmp_path)


def test_external_evaluation_rejects_reported_overlap(tmp_path: Path) -> None:
    make_cohort(tmp_path)
    path = tmp_path / "image_audit.json"
    audit = json.loads(path.read_text())
    audit["overlaps"] = [{"kind": "exact_pixel_overlap"}]
    path.write_text(json.dumps(audit))
    with pytest.raises(ValueError, match="reports overlaps"):
        verified_images(tmp_path)
