"""Protect cross-class grouping and lateral-derived duplicate separation."""

import pytest
from src.data.clean_split import audit_splits
from src.data.prepare_ssmu_split import fixed_folds, group_and_select, source_record


def row(directory: str, stem: int, view: str = "pa", mode: str = "L") -> dict:
    name = f"dataset/images/{directory}/{stem}_{view}.png"
    return source_record(
        {"image_id": name, "mode": mode, "sha256": name, "pixel_sha256": name, "phash": 0}
    )


def pair(first: dict, second: dict, kind: str = "exact_bytes") -> dict:
    return {
        "ssmu_image_id": first["image_id"],
        "reference_source": "SSMU",
        "reference_image_id": second["image_id"],
        "kind": kind,
    }


def test_equal_numbers_join_both_classes_without_merging_images() -> None:
    records = [row("norma", 1), row("pneumonia", 1), row("pneumonia", 2)]
    components, _, selected = group_and_select(records, [])
    assert components.root(0) == components.root(1)
    assert components.root(0) != components.root(2)
    assert selected == [0, 1, 2]
    assert records[0]["label"] == "NORMAL"
    assert all(not r["patient_identity_verified"] for r in records)


def test_lateral_match_keeps_distinct_frontal_images_in_one_group() -> None:
    records = [
        row("norma", 7),
        row("norma", 7, "lat"),
        row("norma", 32),
        row("norma", 32, "lat"),
        row("pneumonia", 32),
    ]
    components, _, selected = group_and_select(records, [pair(records[1], records[3])])
    assert len({components.root(i) for i in range(5)}) == 1
    assert selected == [0, 2, 4]


def test_front_duplicate_retains_lexicographic_representative() -> None:
    records = [row("pneumonia", 4), row("pneumonia", 103)]
    components, _, selected = group_and_select(records, [pair(records[0], records[1])])
    assert selected == [1]
    assert components.root(0) == components.root(1)


def test_conflicting_duplicate_and_highbit_frontal_are_rejected() -> None:
    records = [row("norma", 1), row("pneumonia", 2)]
    with pytest.raises(ValueError, match="Conflicting"):
        group_and_select(records, [pair(*records)])
    with pytest.raises(ValueError, match="8-bit"):
        group_and_select([row("norma", 1, mode="I;16")], [])


def test_unknown_filename_is_not_silently_assigned_a_patient_proxy() -> None:
    with pytest.raises(ValueError, match="layout"):
        source_record({"image_id": "dataset/images/norma/unknown.png"})


def test_fixed_folds_keep_global_groups_and_lateral_exclusions_disjoint() -> None:
    records = [
        row(directory, i, view)
        for i in range(1, 36)
        for directory in ("norma", "pneumonia")
        for view in ("pa", "lat")
    ]
    components, pairs, selected = group_and_select(records, [])
    splits = fixed_folds(records, components, selected)
    assert splits == fixed_folds(records, components, selected)
    audit = audit_splits(records, splits, components, pairs)
    assert len(splits["excluded"]) == 70
    assert all(not any(v.values()) for v in audit["overlaps"].values())
    assert all(
        {records[i]["label_id"] for i in splits[s]} == {0, 1} for s in ("train", "val", "test")
    )
