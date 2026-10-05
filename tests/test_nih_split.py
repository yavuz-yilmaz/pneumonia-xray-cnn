"""Verify that prior exposure and duplicate bridges cannot leak into new holdouts."""

import pytest
from src.data.clean_split import Components
from src.data.prepare_nih_split import assign_nih_splits


def sources() -> list[dict]:
    return [
        {
            "source_family": "nih",
            "nih_patient_id": f"{i + 1:08d}",
            "eligible_adult": True,
            "label_id": int(i >= 24),
            "pixel_sha256": f"pixels-{i}",
        }
        for i in range(48)
    ]


def patient_sets(records: list[dict], splits: dict) -> dict:
    return {
        split: {records[i]["nih_patient_id"] for i in indices} for split, indices in splits.items()
    }


def test_duplicate_bridges_inherit_historical_roles() -> None:
    records = sources()
    roles = {
        "00000001": "external",
        "00000002": "train",
        "00000003": "val",
        "00000007": "train",
        "00000008": "val",
    }
    groups = Components(len(records))
    for first, second in [(0, 3), (1, 5), (2, 4), (6, 7)]:
        groups.join(first, second)
    splits, reasons = assign_nih_splits(records, groups, roles, 20261001)
    assert {0, 3, 6, 7} <= set(splits["excluded"])
    assert {1, 5} <= set(splits["train"])
    assert {2, 4} <= set(splits["val"])
    assert reasons[3] == "touches_reserved_rsna_external_patient"
    assert reasons[7] == "conflicting_historical_roles"
    assert not (patient_sets(records, splits)["test"] & roles.keys())


def test_identical_pixels_with_conflicting_report_labels_are_excluded() -> None:
    records = sources()
    records[25]["pixel_sha256"] = records[10]["pixel_sha256"]
    groups = Components(len(records))
    groups.join(10, 25)
    splits, reasons = assign_nih_splits(records, groups, {}, 20261001)
    assert {10, 25} <= set(splits["excluded"])
    assert reasons[10] == "identical_pixels_with_conflicting_pneumonia_labels"


def test_pediatric_duplicate_cannot_enter_new_test_or_training() -> None:
    records = sources()
    records.append({"source_family": "legacy_pediatric", "pixel_sha256": "legacy"})
    groups = Components(len(records))
    groups.join(0, len(records) - 1)
    splits, reasons = assign_nih_splits(records, groups, {}, 20261001)
    assert 0 in splits["excluded"]
    assert reasons[0] == "touches_observed_pediatric_data"


def test_minor_with_reserved_identity_protects_adult_duplicate() -> None:
    records = sources()
    records[0]["eligible_adult"] = False
    groups = Components(len(records))
    groups.join(0, 1)
    splits, reasons = assign_nih_splits(records, groups, {"00000001": "external"}, 20261001)
    assert {0, 1} <= set(splits["excluded"])
    assert reasons[1] == "touches_reserved_rsna_external_patient"


def test_new_partitions_do_not_depend_on_input_order() -> None:
    original = sources()
    reversed_rows = list(reversed(original))
    first, _ = assign_nih_splits(original, Components(len(original)), {}, 20261001)
    second, _ = assign_nih_splits(reversed_rows, Components(len(reversed_rows)), {}, 20261001)
    assert patient_sets(original, first) == patient_sets(reversed_rows, second)


def test_one_class_split_is_rejected() -> None:
    records = sources()
    for row in records:
        row["label_id"] = 0
    with pytest.raises(ValueError, match="Both labels required"):
        assign_nih_splits(records, Components(len(records)), {}, 20261001)
