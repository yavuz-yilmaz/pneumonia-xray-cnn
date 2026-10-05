"""Check original-patient lineage and age exclusions before adult experiments."""

import pytest
from src.data.inspect_nih import inspect_rows


def row(patient: int, labels: str, age: int = 50, acquisition: int = 0) -> dict:
    return {
        "Image Index": f"{patient:08d}_{acquisition:03d}.png",
        "Patient ID": str(patient),
        "Patient Age": str(age),
        "Finding Labels": labels,
    }


def test_repeated_patient_keeps_exposure_across_acquisitions() -> None:
    rows = [
        row(1, "Pneumonia"),
        row(1, "Effusion|Pneumonia", acquisition=1),
        row(2, "Pneumonia", age=17),
        row(3, "No Finding"),
        row(4, "Pneumonia"),
        row(5, "Pneumonia", age=412),
    ]
    mappings = [{"img_id": rows[0]["Image Index"]}]
    result = inspect_rows(rows, mappings, {"00000001"})
    assert result["adult_pneumonia_images"] == 3
    assert result["adult_pneumonia_patients"] == 2
    assert result["fresh_adult_pneumonia_images"] == 1
    assert result["fresh_adult_pneumonia_patients"] == 1
    assert result["adult_pneumonia_images_in_rsna"] == 1
    assert result["age_exclusions"] == {"under_18": 1, "implausible_age": 1}
    assert all(
        r["previous_rsna_exposure"]
        for r in result["adult_records"]
        if r["patient_id"] == "00000001"
    )


def test_missing_rsna_mapping_is_rejected() -> None:
    with pytest.raises(ValueError, match="cover every RSNA"):
        inspect_rows([row(1, "No Finding")], [{"img_id": "00000002_000.png"}], set())


def test_filename_patient_mismatch_is_rejected() -> None:
    source = row(1, "Pneumonia")
    source["Patient ID"] = "2"
    with pytest.raises(ValueError, match="patient ID"):
        inspect_rows([source], [], set())


def test_conflicting_no_finding_label_is_rejected() -> None:
    with pytest.raises(ValueError, match="Conflicting"):
        inspect_rows([row(1, "No Finding|Pneumonia")], [], set())


def test_duplicate_image_ids_are_rejected() -> None:
    with pytest.raises(ValueError, match="Repeated"):
        inspect_rows([row(1, "Pneumonia"), row(1, "Pneumonia")], [], set())
