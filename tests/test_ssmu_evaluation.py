"""Check one-time scoring and validation gates before reserved-test access."""

import json
from pathlib import Path

import pytest
from src.evaluation.evaluate_ssmu import evaluate, freeze_selection


@pytest.mark.parametrize("prior", ["output", "sentinel"])
def test_repeat_evaluation_is_rejected_before_missing_inputs(tmp_path: Path, prior: str) -> None:
    run, output = tmp_path / "run", tmp_path / "out"
    run.mkdir()
    if prior == "output":
        output.mkdir()
    else:
        (run / "test_started.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="already started"):
        evaluate(run, tmp_path / "no_test_manifest", output)
    assert not (run / "selected_joint").exists()


def test_infeasible_validation_policy_does_not_open_test_or_export(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "result.json").write_text(
        json.dumps(
            {
                "status": "complete_development_only",
                "joint_threshold": None,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="feasible validation"):
        freeze_selection(run, tmp_path / "no_test_manifest", run / "selected_joint")
    assert not (run / "selected_joint").exists()
    assert not (run / "test_started.json").exists()


def test_training_source_drift_blocks_export_before_cohort_access(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    source = tmp_path / "source.py"
    source.write_text("changed", encoding="utf-8")
    (run / "result.json").write_text(
        json.dumps(
            {
                "status": "complete_development_only",
                "joint_threshold": 0.5,
            }
        ),
        encoding="utf-8",
    )
    (run / "protocol.json").write_text(
        json.dumps(
            {
                "input_sha256": {str(source): "invalid"},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="training input changed"):
        freeze_selection(run, tmp_path / "no_test_manifest", run / "selected_joint")
    assert not (run / "selected_joint").exists()
