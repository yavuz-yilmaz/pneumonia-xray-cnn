"""Check source readiness and sequential fail-closed execution gates."""

import json
import subprocess
from pathlib import Path

import pytest
from scripts.run_nih_experiment import execute_stage, experiment_stages, extraction_count


def archive() -> dict:
    return {
        "name": "images_001.tar.gz",
        "url": "https://nihcc.box.com/shared/static/example.gz",
        "expected_bytes": 100,
    }


def test_missing_extraction_is_not_ready(tmp_path: Path) -> None:
    assert extraction_count(tmp_path, [archive()]) == 0


def test_failed_integrity_flag_cannot_start_training(tmp_path: Path) -> None:
    directory = tmp_path / "images_001"
    directory.mkdir()
    (directory / "image_audit.json").write_text(
        json.dumps(
            {
                "archive": "images_001.tar.gz",
                "images": 1,
                "gzip_crc_verified": False,
                "all_image_pixels_decoded": True,
            }
        )
    )
    with pytest.raises(ValueError, match="integrity gate"):
        extraction_count(tmp_path, [archive()])


def test_final_test_is_last_and_follows_both_registered_candidates() -> None:
    stages = experiment_stages()
    assert [name for name, _ in stages] == [
        "split",
        "resnet18",
        "efficientnet_v2_s",
        "select",
        "test",
    ]
    assert stages[-1][1] == ["src.evaluation.evaluate_nih", "test"]
    for _, command in stages[1:3]:
        assert command[command.index("--minimum-recall") + 1] == "0.90"


def test_failed_stage_propagates_and_preserves_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failed(command: list, **kwargs: object) -> None:
        assert kwargs["check"] is True
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr("scripts.run_nih_experiment.subprocess.run", failed)
    with pytest.raises(subprocess.CalledProcessError):
        execute_stage("split", ["src.data.prepare_nih_split"], tmp_path)
    assert (tmp_path / "split.log").exists()
