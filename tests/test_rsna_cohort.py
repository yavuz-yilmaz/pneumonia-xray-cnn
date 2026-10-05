from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from src.data.clean_split import fingerprint
from src.data.extract_rsna_cohort import image_signature
from src.data.prepare_rsna_cohort import cohort_records


def metadata() -> tuple[list[dict], dict]:
    mappings, rows = [], []
    for patient in range(4):
        for acquisition in range(2):
            sop = f"{patient}.{acquisition}"
            mappings.append(
                {
                    "SOPInstanceUID": sop,
                    "img_id": f"{patient:08d}_{acquisition:03d}.png",
                    "subset_img_id": sop,
                }
            )
            rows.append({"SOPInstanceUID": sop, "labelId": "normal" if patient < 2 else "opacity"})
    annotations = {
        "labelGroups": [
            {
                "name": "Calculated",
                "labels": [
                    {"id": "normal", "name": "Normal"},
                    {"id": "opacity", "name": "Lung Opacity"},
                ],
            }
        ],
        "datasets": [{"annotations": rows}],
    }
    return mappings, annotations


def test_cohort_is_patient_distinct_and_independent_of_source_order() -> None:
    mappings, annotations = metadata()
    cohort = cohort_records(mappings, annotations, 2, 42)
    annotations["datasets"][0]["annotations"].reverse()
    assert cohort == cohort_records(list(reversed(mappings)), annotations, 2, 42)
    assert len(cohort) == len({row["nih_patient_id"] for row in cohort}) == 4
    assert [row["label_id"] for row in cohort] == [0, 0, 1, 1]


def test_cohort_rejects_conflicting_final_annotations() -> None:
    mappings, annotations = metadata()
    annotations["datasets"][0]["annotations"].append(
        {"SOPInstanceUID": "0.0", "labelId": "opacity"}
    )
    with pytest.raises(ValueError, match="Conflicting final labels"):
        cohort_records(mappings, annotations, 2, 42)


def test_reserved_patients_are_excluded_with_all_acquisitions() -> None:
    mappings, annotations = metadata()
    cohort = cohort_records(
        mappings, annotations, 1, 42, excluded_patients={"00000000", "00000002"}
    )
    assert {row["nih_patient_id"] for row in cohort} == {"00000001", "00000003"}


def test_external_fingerprint_matches_pediatric_audit(tmp_path: Path) -> None:
    directory = tmp_path / "train" / "NORMAL"
    directory.mkdir(parents=True)
    path = directory / "IM-0001-0001.png"
    image = Image.fromarray(np.random.default_rng(42).integers(0, 256, (80, 96), dtype=np.uint8))
    image.save(path)
    expected = fingerprint(path)
    pixel_hash, phash, thumbnail = image_signature(image)
    assert pixel_hash == expected["pixel_sha256"]
    assert phash == expected["phash"]
    np.testing.assert_array_equal(thumbnail, expected["thumbnail"])
