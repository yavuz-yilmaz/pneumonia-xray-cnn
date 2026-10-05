"""Check pairing, undefined precision and repeat protection for the exploratory check."""

from pathlib import Path

import numpy as np
import pytest
from src.evaluation import evaluate_ssmu_openi as external


def test_identical_models_have_zero_paired_differences() -> None:
    """Identical predictions must have identical metrics on each sampled case set."""
    labels = np.array([0, 0, 0, 1, 1, 1])
    scores = {name: np.array([0.1, 0.3, 0.7, 0.2, 0.6, 0.9]) for name in external.NAMES}
    result = external.paired_intervals(labels, scores, dict.fromkeys(scores, 0.5), 30)
    for comparison in result["ssmu_minus_comparator"].values():
        for metric in comparison.values():
            assert metric["interval_95"] == [0.0, 0.0]
            assert metric["valid_repetitions"] > 0


def test_no_positive_predictions_do_not_invent_precision() -> None:
    """Preserve undefined precision instead of assigning a misleading perfect score."""
    scores = {name: np.zeros(4) for name in external.NAMES}
    result = external.paired_intervals(
        np.array([0, 0, 1, 1]), scores, dict.fromkeys(scores, 0.5), 20
    )
    assert result["models"]["ssmu"]["precision"] == {
        "valid_repetitions": 0,
        "interval_95": None,
    }


@pytest.mark.parametrize("threshold", [np.nan, -0.1, 1.1])
def test_invalid_threshold_is_rejected(threshold: float) -> None:
    """A bad operating point cannot silently produce a comparison."""
    scores = {name: np.zeros(4) for name in external.NAMES}
    with pytest.raises(ValueError, match="threshold"):
        external.paired_intervals(
            np.array([0, 0, 1, 1]), scores, dict.fromkeys(scores, threshold), 20
        )


def test_existing_run_is_preserved_before_inputs_or_inference(tmp_path: Path) -> None:
    """An existing or failed study cannot be overwritten by a repeat invocation."""
    output = tmp_path / "already_registered"
    output.mkdir()
    with pytest.raises(FileExistsError, match="preserving"):
        external.evaluate(tmp_path, output)
