"""Shared binary operating-point selection using validation predictions only."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def binary_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    """Report positive-class performance and negative-class specificity."""
    predictions = probabilities >= threshold
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "sample_count": len(labels),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "specificity": float(tn / max(tn + fp, 1)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "average_precision": float(average_precision_score(labels, probabilities)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def select_threshold(
    labels: np.ndarray, probabilities: np.ndarray, minimum_recall: float = 0.99
) -> float:
    """Maximize validation specificity subject to a prespecified recall floor.

    Ties prefer higher recall then a threshold closer to .5. No test labels or
    test probabilities may be passed to this function for model selection.
    """
    if not 0 < minimum_recall <= 1:
        raise ValueError("minimum_recall must be in (0, 1]")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Validation must contain both classes")
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Probabilities must be finite and in [0, 1]")
    fpr, tpr, thresholds = roc_curve(labels, probabilities, drop_intermediate=False)
    eligible = np.flatnonzero((tpr >= minimum_recall) & np.isfinite(thresholds))
    best = min(eligible, key=lambda i: (fpr[i], -tpr[i], abs(thresholds[i] - 0.5)))
    return float(thresholds[best])


def select_f1_threshold(
    labels: np.ndarray, probabilities: np.ndarray, minimum_recall: float = 0.9
) -> float:
    """Maximize validation F1 under a fixed recall floor for imbalanced NIH labels."""
    # Reuse the existing input validation; its selected threshold is not used.
    select_threshold(labels, probabilities, minimum_recall)
    fpr, tpr, thresholds = roc_curve(labels, probabilities, drop_intermediate=False)
    positives = np.count_nonzero(labels == 1)
    negatives = len(labels) - positives
    tp, fp, fn = tpr * positives, fpr * negatives, (1 - tpr) * positives
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    eligible = np.flatnonzero((tpr >= minimum_recall) & np.isfinite(thresholds))
    best = min(eligible, key=lambda i: (-f1[i], fpr[i], -tpr[i], abs(thresholds[i] - 0.5)))
    return float(thresholds[best])
