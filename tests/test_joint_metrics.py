"""Check inclusive threshold ties, baseline floors and impossible improvements."""

import numpy as np
import pytest
from src.evaluation.decision import binary_metrics
from src.training.calibrate_joint_metrics import select_joint_threshold


def reference(value: float) -> dict:
    return {
        name: dict.fromkeys(("accuracy", "precision", "recall", "f1"), value)
        for name in ("original", "current")
    }


def test_perfect_separation_improves_all_four() -> None:
    labels = np.array([0, 1, 0, 1])
    scores = np.array([0.1, 0.8, 0.2, 0.9])
    assert select_joint_threshold(labels, scores, reference(0.8)) == 0.8


def test_impossible_or_equal_reference_has_no_fallback() -> None:
    labels = np.array([0, 1, 0, 1])
    assert select_joint_threshold(labels, np.array([0.9, 0.8, 0.7, 0.6]), reference(1)) is None
    assert select_joint_threshold(labels, np.array([0.1, 0.8, 0.2, 0.9]), reference(1)) is None


def test_tied_probabilities_are_classified_together() -> None:
    labels = np.array([0, 1, 0, 1])
    scores = np.array([0.1, 0.8, 0.8, 0.9])
    baselines = reference(0.4)
    baselines["current"]["recall"] = 1.0
    threshold = select_joint_threshold(labels, scores, baselines)
    assert threshold == 0.8
    assert binary_metrics(labels, scores, threshold)["recall"] == 1.0


def test_each_metric_uses_stronger_of_both_references() -> None:
    labels = np.array([0, 1, 0, 1])
    scores = np.array([0.1, 0.8, 0.8, 0.9])
    baselines = reference(0.4)
    baselines["original"]["recall"] = 1.0
    baselines["current"]["precision"] = 1.0
    assert select_joint_threshold(labels, scores, baselines) is None


@pytest.mark.parametrize("scores", [np.array([0.1, np.nan]), np.array([0.1, 1.1])])
def test_invalid_scores_rejected(scores: np.ndarray) -> None:
    with pytest.raises(ValueError, match="Invalid joint"):
        select_joint_threshold(np.array([0, 1]), scores, reference(0.2))


def test_vectorized_selector_matches_exhaustive_metrics_with_many_ties() -> None:
    rng = np.random.default_rng(20261001)
    for _ in range(10):
        labels = np.r_[np.zeros(28, dtype=int), np.ones(12, dtype=int)]
        rng.shuffle(labels)
        scores = np.round(rng.uniform(size=len(labels)), 1)
        baselines = reference(0.1)
        baselines["original"]["recall"] = 0.4
        floors = {key: max(row[key] for row in baselines.values()) for key in baselines["original"]}
        feasible = []
        for threshold in np.unique(scores):
            metrics = binary_metrics(labels, scores, float(threshold))
            if all(metrics[key] >= value - 1e-12 for key, value in floors.items()) and any(
                metrics[key] > value + 1e-12 for key, value in floors.items()
            ):
                feasible.append((metrics["f1"], metrics["accuracy"], float(threshold)))
        expected = max(feasible)[2] if feasible else None
        assert select_joint_threshold(labels, scores, baselines) == expected
