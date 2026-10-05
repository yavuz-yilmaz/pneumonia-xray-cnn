"""Meaningful overlap, aggregation, provenance and one-time external scoring checks."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from src.data.download_nih import file_sha256
from src.evaluation import evaluate_openi as external


def case(case_id: str, positive: bool, ids: list[str]) -> dict:
    """Make an explicitly eligible metadata-only case."""
    return {
        "case_id": case_id,
        "metadata_eligible": True,
        "exclusion_reasons": [],
        "candidate_label": "MANUAL_REPORT_PNEUMONIA"
        if positive
        else "NO_CODED_OR_MENTIONED_PNEUMONIA",
        "image_ids": ids,
    }


def image(
    image_id: str, case_id: str, view: str = "Frontal", pixel_hash: str | None = None
) -> dict:
    """Make a decoded image record without any prediction fields."""
    return {
        "image_id": image_id,
        "case_id": case_id,
        "secondary_projection": view,
        "pixel_sha256": pixel_hash or image_id,
    }


def test_overlap_on_lateral_removes_entire_case_and_cross_case_pairs() -> None:
    """A lateral duplicate can leak a patient's frontal image as well."""
    cases = [
        case("p", True, ["p_f"]),
        case("n", False, ["n_f"]),
        case("leak", True, ["leak_f", "leak_l"]),
        case("a", False, ["a_f"]),
        case("b", False, ["b_f"]),
    ]
    images = [
        image("p_f", "p"),
        image("n_f", "n"),
        image("leak_f", "leak"),
        image("leak_l", "leak", "Lateral"),
        image("a_f", "a"),
        image("b_f", "b"),
    ]
    matches = [
        {"openi_image_id": "leak_l", "openi_case_id": "leak", "reference_source": "NIH"},
        {
            "openi_image_id": "a_f",
            "openi_case_id": "a",
            "reference_source": "OpenI",
            "reference_image_id": "b_f",
            "reference_case_id": "b",
        },
    ]
    retained, excluded = external.derive_cohort(cases, images, matches)
    assert {r["case_id"] for r in retained} == {"p", "n"}
    reasons = {r["case_id"]: r["reasons"] for r in excluded}
    assert reasons["leak"] == ["cross_source_duplicate"]
    assert reasons["a"] == reasons["b"] == ["cross_case_duplicate"]


def test_unique_frontals_and_missing_unmapped_metadata_are_handled_before_scoring() -> None:
    """Prevent duplicated frames and unmapped images from changing case weight."""
    cases = [
        case("p", True, ["p1", "p2", "p3"]),
        case("n", False, ["n1"]),
        case("unknown", False, ["u1"]),
        case("missing", True, ["missing_image"]),
    ]
    images = [
        image("p1", "p", pixel_hash="same"),
        image("p2", "p", pixel_hash="same"),
        image("p3", "p", "Lateral"),
        image("n1", "n"),
        image("u1", "unknown", None),
    ]
    retained, excluded = external.derive_cohort(cases, images, [])
    assert [r["image_id"] for r in next(r for r in retained if r["case_id"] == "p")["images"]] == [
        "p1"
    ]
    assert {r["case_id"] for r in excluded} == {"unknown", "missing"}


def test_mismatched_duplicate_owner_is_refused() -> None:
    """Reject duplicate reports that attach an image to another patient case."""
    with pytest.raises(ValueError, match="ownership"):
        external.derive_cohort(
            [case("p", True, ["p1"])],
            [image("p1", "p")],
            [{"openi_image_id": "p1", "openi_case_id": "different", "reference_source": "NIH"}],
        )


def test_aggregate_gives_one_prediction_per_case_before_thresholding() -> None:
    """Two images from one case must not count as two independent patients."""
    rows = [
        {"case_id": "p", "label_id": 1},
        {"case_id": "n", "label_id": 0},
        {"case_id": "p", "label_id": 1},
    ]
    ids, targets, scores = external.aggregate_cases(
        rows, np.array([1, 0, 1]), np.array([0.2, 0.3, 0.8])
    )
    assert ids == ["n", "p"]
    np.testing.assert_array_equal(targets, [0, 1])
    np.testing.assert_allclose(scores, [0.3, 0.5])
    assert external.case_metrics(targets, scores, 0.5)["recall"] == 1


@pytest.mark.parametrize("scores", [np.array([np.nan, 0.3]), np.array([0.2, 1.2])])
def test_invalid_predictions_refused(scores: np.ndarray) -> None:
    """Bad model outputs cannot produce a favorable metric."""
    with pytest.raises(ValueError, match="probabilities"):
        external.aggregate_cases(
            [{"case_id": "a", "label_id": 0}, {"case_id": "b", "label_id": 1}],
            np.array([0, 1]),
            scores,
        )


def test_paired_identical_models_have_zero_differences() -> None:
    """Independent resamples per model would fail this exact paired invariant."""
    labels = np.array([0, 0, 0, 1, 1, 1])
    scores = {name: np.array([0.1, 0.3, 0.7, 0.2, 0.6, 0.9]) for name in external.MODEL_NAMES}
    intervals = external.paired_intervals(labels, scores, dict.fromkeys(scores, 0.5), 40)
    for comparator in intervals["mimic_minus_comparator"].values():
        for result in comparator.values():
            assert result["interval_95"] == [0.0, 0.0]
            assert result["valid_repetitions"] > 0


def test_no_positive_prediction_has_explicitly_undefined_precision() -> None:
    """A model that predicts no disease must not earn invented precision evidence."""
    labels = np.array([0, 0, 1, 1])
    scores = {name: np.zeros(4) for name in external.MODEL_NAMES}
    assert external.case_metrics(labels, scores["mimic"], 0.5)["precision"] is None
    intervals = external.paired_intervals(labels, scores, dict.fromkeys(scores, 0.5), 20)
    assert intervals["models"]["mimic"]["precision"] == {
        "valid_repetitions": 0,
        "interval_95": None,
    }
    assert intervals["models"]["mimic"]["recall"]["interval_95"] == [0.0, 0.0]


def test_checkpoint_drift_refused_before_deserialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Even a local checkpoint must match its pinned input before loading."""
    path = tmp_path / "model.pt"
    path.write_bytes(b"changed")
    monkeypatch.setattr(external.torch, "load", lambda *a, **kw: pytest.fail("deserialized drift"))
    with pytest.raises(ValueError, match="changed before loading"):
        external.read_model("original", {"path": str(path), "sha256": "incorrect"})


def test_unverified_xrv_checkpoint_cannot_score_openi(tmp_path: Path) -> None:
    """A mixed pretraining architecture cannot bypass source provenance gates."""
    path = tmp_path / "model.pt"
    torch.save({"label_mapping": external.LABEL_TO_ID, "model_name": "xrv_densenet121"}, path)
    with pytest.raises(ValueError, match="Unverified XRV"):
        external.read_model("current", {"path": str(path), "sha256": file_sha256(path)})


def test_source_threshold_and_input_contract_are_checked(tmp_path: Path) -> None:
    """Preserve the original operating point and reject invalid normalization."""
    path = tmp_path / "model.pt"
    saved = {
        "label_mapping": external.LABEL_TO_ID,
        "model_name": "resnet18",
        "image_size": 224,
        "decision_threshold": 0.8,
    }
    torch.save(saved, path)
    with pytest.raises(ValueError, match="threshold"):
        external.read_model("original", {"path": str(path), "sha256": file_sha256(path)})
    saved["decision_threshold"] = 0.7
    saved["normalization"] = {"mean": [0.0, 0.0, 0.0], "std": [1.0, 0.0, 1.0]}
    torch.save(saved, path)
    with pytest.raises(ValueError, match="input contract"):
        external.read_model("original", {"path": str(path), "sha256": file_sha256(path)})


def test_existing_evaluation_is_preserved_before_any_read(tmp_path: Path) -> None:
    """No automatic second look at a scored or failed cohort."""
    output = tmp_path / "evaluation"
    output.mkdir()
    with pytest.raises(FileExistsError, match="already registered"):
        external.evaluate(tmp_path, tmp_path / "missing", output)


def test_failed_inference_keeps_marker_and_blocks_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failure after registration stays visible and cannot silently rescore."""
    cohort = tmp_path / "cohort"
    cohort.mkdir()
    source = tmp_path / "image.png"
    source.write_bytes(b"fixture bytes; no real external image")
    with (cohort / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["filepath", "label_id", "case_id", "sha256"])
        writer.writeheader()
        writer.writerow(
            {
                "filepath": source.as_posix(),
                "label_id": 1,
                "case_id": "p",
                "sha256": file_sha256(source),
            }
        )
    (cohort / "cohort_audit.json").write_text(
        json.dumps(
            {
                "artifact_sha256": {"manifest.csv": file_sha256(cohort / "manifest.csv")},
                "input_sha256": {},
                "source_sha256": {},
            }
        )
    )
    monkeypatch.setattr(
        external, "model_inputs", lambda root: dict.fromkeys(external.MODEL_NAMES, {})
    )
    monkeypatch.setattr(external, "read_model", lambda *a: {"decision_threshold": 0.5})

    def fail(*args: object) -> tuple:
        raise RuntimeError("fixture inference failed")

    monkeypatch.setattr(external, "predict_images", fail)
    output = tmp_path / "evaluation"
    with pytest.raises(RuntimeError, match="fixture inference failed"):
        external.evaluate(tmp_path, cohort, output, "cpu")
    assert json.loads((output / "status.json").read_bytes())["state"] == "failed_preserved"
    assert (output / "registration.json").is_file()
    with pytest.raises(FileExistsError):
        external.evaluate(tmp_path, cohort, output, "cpu")


def write_json(path: Path, value: dict) -> None:
    """Write local fixture metadata without borrowing any external patient data."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def registered_models(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Provide a full local selection/provenance contract with tiny mock artifacts."""
    baseline_hashes = {}
    for name, relative in [
        ("original", "models/best_model.pt"),
        ("current", "models/clean_v3/selected_matched98/best_model.pt"),
        ("nih", "models/nih_v1/selected/best_model.pt"),
    ]:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
        baseline_hashes[name] = file_sha256(path)
    monkeypatch.setattr(
        external, "BASELINE_HASHES", {k: v for k, v in baseline_hashes.items() if k != "nih"}
    )
    source = tmp_path / "src/training/train_xray_probe.py"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"fixed training source fixture")
    write_json(tmp_path / "reports/metrics/nih_run_v1/status.json", {"state": "complete"})
    write_json(
        tmp_path / "reports/metrics/nih_mimic_run_v1/status.json",
        {"state": "complete_train_validation_only"},
    )
    write_json(
        tmp_path / "reports/metrics/nih_mimic_run_v1/registration.json",
        {
            "prerequisite_test_started_at_registration": False,
            "source_sha256": {"src/training/train_xray_probe.py": file_sha256(source)},
        },
    )
    manifests = {"train": "train hash", "val": "val hash", "test": "test hash"}
    write_json(
        tmp_path / "models/nih_v1/selected/selection.json",
        {
            "selected_checkpoint_sha256": baseline_hashes["nih"],
            "test_used_for_selection": False,
            "data_audit": {"manifest_sha256": manifests},
        },
    )
    candidates = []
    for regularization in [0.001, 0.01, 0.1, 1.0]:
        relative = f"models/nih_mimic_v1/frozen/c{regularization}"
        directory = tmp_path / relative
        directory.mkdir(parents=True)
        (directory / "best_model.pt").write_bytes(b"fixture model, never deserialized")
        validation = {"average_precision": 0.5 if regularization in {0.01, 0.1} else 0.3}
        candidates.append({"directory": relative, "C": regularization, "validation": validation})
        write_json(
            directory / "protocol.json",
            {
                "pretraining": "mimic_ch",
                "test_access": "none",
                "backbone_training": "frozen",
                "external_weights_sha256": external.MIMIC_SOURCE_SHA256,
                "task": "nih_report_pneumonia",
                "C": regularization,
                "minimum_recall": 0.9,
                "source_sha256": file_sha256(source),
                "data_audit": {"manifest_sha256": manifests},
                "pretraining_metadata": {
                    "weights": "densenet121-res224-mimic_ch",
                    "source_sha256": external.MIMIC_SOURCE_SHA256,
                    "feature_checkpoint_sha256": external.MIMIC_FEATURE_SHA256,
                },
            },
        )
    write_json(
        tmp_path / "models/nih_mimic_v1/frozen/result.json",
        {
            "status": "complete",
            "candidates": candidates,
            "selected_candidate": candidates[1],
        },
    )
    return tmp_path


def test_source_selection_is_ap_then_lowest_c(registered_models: Path) -> None:
    """Tie resolution must match the training-only prospective recipe."""
    records = external.model_inputs(registered_models)
    assert Path(records["mimic"]["path"]).parent.name == "c0.01"
    assert set(records) == external.MODEL_NAMES


def test_changed_selection_cannot_replace_registered_winner(registered_models: Path) -> None:
    """A later test-driven substitution must fail even if its file exists."""
    path = registered_models / "models/nih_mimic_v1/frozen/result.json"
    result = json.loads(path.read_bytes())
    result["selected_candidate"] = result["candidates"][2]
    write_json(path, result)
    with pytest.raises(ValueError, match="fixed validation selection"):
        external.model_inputs(registered_models)


@pytest.mark.parametrize(
    "violation", ["mixed_source", "different_split", "changed_baseline", "unfinished"]
)
def test_provenance_split_and_prerequisites_cannot_be_bypassed(
    registered_models: Path, violation: str
) -> None:
    """Refuse source contamination and incompatible comparisons before inference."""
    if violation == "changed_baseline":
        (registered_models / "models/best_model.pt").write_bytes(b"replacement")
    elif violation == "unfinished":
        write_json(
            registered_models / "reports/metrics/nih_mimic_run_v1/status.json",
            {"state": "training"},
        )
    else:
        path = registered_models / "models/nih_mimic_v1/frozen/c0.01/protocol.json"
        protocol = json.loads(path.read_bytes())
        if violation == "mixed_source":
            protocol["pretraining_metadata"]["weights"] = "densenet121-res224-all"
        else:
            protocol["data_audit"]["manifest_sha256"]["train"] = "replacement train split"
        write_json(path, protocol)
    with pytest.raises(ValueError):
        external.model_inputs(registered_models)
