"""Check the prospective NIH threshold criterion on explicit confusion counts."""

import numpy as np
import pytest
from src.evaluation.decision import binary_metrics, select_f1_threshold, select_threshold


def test_f1_can_prefer_more_recall_than_the_specificity_criterion() -> None:
    labels = np.array([1, 0, 1, 0])
    probabilities = np.array([0.9, 0.6, 0.5, 0.2])
    legacy = select_threshold(labels, probabilities, minimum_recall=0.5)
    nih = select_f1_threshold(labels, probabilities, minimum_recall=0.5)
    assert legacy == 0.9
    assert nih == 0.5
    assert binary_metrics(labels, probabilities, nih)["f1"] == pytest.approx(0.8)


def test_recall_floor_is_preserved_even_when_unconstrained_f1_is_higher() -> None:
    labels = np.array([1, 0, 0, 0, 0, 1])
    probabilities = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.1])
    threshold = select_f1_threshold(labels, probabilities, minimum_recall=0.9)
    assert threshold == 0.1
    assert binary_metrics(labels, probabilities, threshold)["recall"] == 1.0


def test_threshold_rejects_invalid_probabilities() -> None:
    with pytest.raises(ValueError, match="Probabilities"):
        select_f1_threshold(np.array([0, 1]), np.array([0.1, np.nan]))
