"""External labels, overlap exclusions and resource reservations must fail closed."""

from __future__ import annotations

import importlib.util
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image
from src.cloud.dinov3_external import (
    MONTGOMERY_BASE,
    display_image,
    external_id,
    external_stage_cost,
    metrics,
    screen,
    validate_selection,
)
from src.cloud.dinov3_job import digest, reserve_cost
from src.data.extract_rsna_cohort import image_signature


@pytest.fixture
def config() -> dict:
    return json.loads(Path("configs/modal_dinov3.json").read_bytes())


def test_external_reservations_preserve_existing_budget(tmp_path: Path, config: dict) -> None:
    path = tmp_path / "ledger.json"
    previous = {
        "ceiling_usd": 24.0,
        "reserved_usd": 23.5853,
        "entries": [{"stage": "previous", "reserved_usd": 23.5853}],
    }
    path.write_text(json.dumps(previous))
    stages = ["external_coordinator", "external_prepare", "external_coordinator", "external_test"]
    assert sum(external_stage_cost(config, s) for s in stages) == pytest.approx(0.386)
    for stage in stages:
        reserve_cost(path, config, stage, "external")
    final = json.loads(path.read_bytes())
    assert final["entries"][0] == previous["entries"][0]
    assert final["reserved_usd"] == pytest.approx(23.9713)
    with pytest.raises(ValueError, match="Budget exhausted"):
        reserve_cost(path, config, "external_test", "duplicate")


def test_positive_only_cohort_reports_sensitivity_without_binary_accuracy() -> None:
    result = metrics([1, 1, 1], [0.9, 0.3, 0.8], 0.6127732396125793)
    assert result["recall"]["value"] == pytest.approx(2 / 3)
    assert result["fn"] == 1
    for name in ("accuracy", "specificity", "precision", "f1", "roc_auc", "average_precision"):
        assert result[name] is None
    assert result["recall"]["wilson95"][0] < 2 / 3 < result["recall"]["wilson95"][1]


def test_binary_metrics_use_frozen_threshold_and_both_classes() -> None:
    result = metrics([0, 0, 1, 1], [0.1, 0.7, 0.6, 0.9], 0.6127732396125793)
    assert [result[k] for k in ("tn", "fp", "fn", "tp")] == [1, 1, 1, 1]
    assert result["accuracy"]["value"] == 0.5
    assert result["f1"] == 0.5
    with pytest.raises(ValueError, match="Invalid"):
        metrics([1], [float("nan")], 0.6)


def test_normal_only_controls_report_specificity_without_auc() -> None:
    result = metrics([0, 0, 0], [0.2, 0.4, 0.8], 0.6127732396125793)
    assert result["specificity"]["value"] == pytest.approx(2 / 3)
    assert result["recall"] is None and result["roc_auc"] is None
    assert result["accuracy"] is None and result["fp"] == 1


def test_original_12_bit_pixels_do_not_clip_to_white() -> None:
    image = Image.fromarray(np.array([[0, 2048, 4095]], dtype=np.uint16))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    result, audit = display_image(buffer.getvalue(), "montgomery_normal")
    assert np.asarray(result)[0, :, 0].tolist() == [0, 128, 255]
    assert "12-bit" in audit["display_conversion"]


def test_supplement_rejects_tb_images_and_partial_source_cohorts() -> None:
    payload = {
        "protocol": {
            "job_id": "1b73b1facc83fda1",
            "threshold": 0.6127732396125793,
            "ssmu2026": {"files": [{"name": "a_pa.png"}]},
        },
        "references": [],
    }
    prepared = {
        "records": [{"key": "a", "source_name": "a_pa.png", "dataset": "ssmu2026"}],
        "errors": {},
    }
    supplement = {
        "dataset": "montgomery_normal",
        "task": "specificity only",
        "files": [
            {
                "name": f"MCUCXR_{i:04d}_0.png",
                "size": 10,
                "url": MONTGOMERY_BASE + f"MCUCXR_{i:04d}_0.png",
            }
            for i in range(80)
        ],
    }
    selection = {
        "external_id": external_id(payload),
        "records": [{"key": "a", "label": 1}],
        "supplement": supplement,
    }
    assert len(validate_selection(payload, prepared, selection)) == 1
    supplement["files"][0]["name"] = "MCUCXR_0000_1.png"
    with pytest.raises(ValueError, match="Invalid Montgomery normal"):
        validate_selection(payload, prepared, selection)
    selection["records"] = []
    with pytest.raises(ValueError, match="Incomplete SSMU"):
        validate_selection(payload, prepared, selection)


def test_unreviewed_rsua_layout_cannot_score_masks_or_overlays() -> None:
    payload = {
        "protocol": {"job_id": "1b73b1facc83fda1", "threshold": 0.6127732396125793},
        "references": [],
    }
    prepared = {"records": [{"key": "a", "source_name": "mask.bmp", "dataset": "rsua"}]}
    selection = {"external_id": external_id(payload), "records": [{"key": "a", "label": 1}]}
    with pytest.raises(ValueError, match="layout has not been reviewed"):
        validate_selection(payload, prepared, selection)


def test_overlap_screen_excludes_prior_and_internal_duplicates_before_predictions() -> None:
    image = Image.fromarray(np.random.default_rng(2).integers(0, 256, (80, 80), dtype=np.uint8))
    pixel_hash, phash, _ = image_signature(image)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    data = buffer.getvalue()
    rows = [{"key": str(i), "sha256": digest(data)} for i in range(2)]
    blobs = {r["key"]: data for r in rows}
    kept, excluded = screen(rows, blobs, [])
    assert len(kept) == 1 and excluded[0]["reason"] == "exact_overlap"
    refs = [{"sha256": "different", "pixel_sha256": pixel_hash, "phash": phash}]
    assert not screen(rows, blobs, refs)[0]
    refs = [{"sha256": "different", "pixel_sha256": "different", "phash": phash ^ 3}]
    assert screen(rows[:1], blobs, refs)[1][0]["reason"] == "conservative_near_overlap"


def test_coordinator_funds_before_external_dispatch_and_reuses_preparation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("modal")
    name = "_test_dinov3_external_coordinator"
    spec = importlib.util.spec_from_file_location(name, "scripts/modal_train_dinov3.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "REMOTE_ROOT", tmp_path)
    monkeypatch.setattr(module, "volume", SimpleNamespace(reload=lambda: None, commit=lambda: None))
    path = tmp_path / "budget_ledger.json"
    path.write_text(
        json.dumps(
            {"ceiling_usd": 24, "reserved_usd": 23.5853, "entries": [{"reserved_usd": 23.5853}]}
        )
    )
    payload = {
        "protocol": {"job_id": "1b73b1facc83fda1", "threshold": 0.6127732396125793},
        "references": [],
    }
    calls = []

    def prepare(payload: dict, evaluation_id: str) -> dict:
        ledger = json.loads(path.read_bytes())
        assert ledger["entries"][-1]["stage"] == "external_prepare"
        directory = tmp_path / "external" / evaluation_id
        directory.mkdir(parents=True)
        result = {"external_id": evaluation_id, "predictions_computed": False}
        (directory / "prepared.json").write_text(json.dumps(result))
        calls.append("prepare")
        return result

    monkeypatch.setattr(module, "external_prepare_remote", SimpleNamespace(remote=prepare))
    result = module.orchestrate.local("external-prepare", payload)
    assert result["external_id"] == external_id(payload)
    module.orchestrate.local("external-prepare", payload)
    assert calls == ["prepare"]
    with pytest.raises(ValueError, match="reviewed external selection"):
        module.orchestrate.local("external-test", payload)


def test_completed_external_result_cannot_be_reused_for_changed_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("modal")
    name = "_test_dinov3_external_locked_result"
    spec = importlib.util.spec_from_file_location(name, "scripts/modal_train_dinov3.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "REMOTE_ROOT", tmp_path)
    monkeypatch.setattr(module, "volume", SimpleNamespace(reload=lambda: None))
    source = {
        "protocol": {"job_id": "1b73b1facc83fda1", "threshold": 0.6127732396125793},
        "references": [],
    }
    selection = {"records": [{"key": "original", "label": 0}]}
    directory = tmp_path / "external" / external_id(source)
    directory.mkdir(parents=True)
    (directory / "result.json").write_text(json.dumps({"complete": True}))
    (directory / "locked_inference.json").write_text(
        json.dumps({"pre_prediction_amendment": selection})
    )
    assert module.orchestrate.local("external-test", {**source, "selection": selection}) == {
        "complete": True
    }
    selection["records"][0]["label"] = 1
    with pytest.raises(ValueError, match="Completed external selection"):
        module.orchestrate.local("external-test", {**source, "selection": selection})
    assert not (tmp_path / "budget_ledger.json").exists()
