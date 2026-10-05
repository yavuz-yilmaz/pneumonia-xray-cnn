"""Evaluate the best trained model on the held-out test split."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.core.config import ConfigFileError, ProjectConfig, load_and_prepare_config
from src.data.dataset import ID_TO_LABEL, LABEL_TO_ID, ChestXRayDataset, XRayDatasetError
from src.data.image_preparation import CheckpointImageTransform
from src.data.transforms import IMAGENET_NORMALIZATION_MEAN, IMAGENET_NORMALIZATION_STD
from src.inference.predict import DEFAULT_PNEUMONIA_THRESHOLD
from src.training.models import build_model
from src.training.train import resolve_device

POSITIVE_LABEL_ID = LABEL_TO_ID["PNEUMONIA"]
NEGATIVE_LABEL_ID = LABEL_TO_ID["NORMAL"]
CLASS_LABELS = [ID_TO_LABEL[NEGATIVE_LABEL_ID], ID_TO_LABEL[POSITIVE_LABEL_ID]]
MEDICAL_WARNING = (
    "This project is for educational purposes; its outputs must not be used for medical diagnosis."
)


@dataclass(frozen=True)
class CheckpointBundle:
    """Loaded model and metadata from a training checkpoint.

    Attributes:
        model: PyTorch model with trained weights loaded.
        metadata: Raw checkpoint metadata excluding no fields.
        checkpoint_path: Path of the loaded checkpoint.
    """

    model: nn.Module
    metadata: dict[str, Any]
    checkpoint_path: Path


@dataclass(frozen=True)
class PredictionBatch:
    """All test predictions and probabilities needed for metric reporting.

    Attributes:
        labels: Ground-truth numeric labels.
        predictions: Predicted numeric labels.
        probabilities: PNEUMONIA probabilities.
    """

    labels: list[int]
    predictions: list[int]
    probabilities: list[float]


@dataclass(frozen=True)
class EvaluationOutputs:
    """Filesystem outputs produced by the evaluation workflow.

    Attributes:
        metrics_path: JSON metrics report path.
        summary_path: Markdown evaluation summary path.
        confusion_matrix_path: Confusion matrix figure path.
        roc_curve_path: ROC curve figure path.
        precision_recall_curve_path: Precision-recall curve figure path.
        misclassified_examples_path: Misclassified image grid path.
    """

    metrics_path: Path
    summary_path: Path
    confusion_matrix_path: Path
    roc_curve_path: Path
    precision_recall_curve_path: Path
    misclassified_examples_path: Path


def load_checkpoint_model(checkpoint_path: Path, device: torch.device) -> CheckpointBundle:
    """Load a trained model checkpoint for evaluation.

    Args:
        checkpoint_path: Path to a checkpoint created by the training workflow.
        device: Runtime device for inference.

    Returns:
        Loaded model and checkpoint metadata.

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
        RuntimeError: If the checkpoint is malformed or incompatible with the model.
    """
    if not checkpoint_path.is_file():
        message = (
            f"Trained model checkpoint file not found: {checkpoint_path}. "
            "First run `python -m src.training.train --config configs/config.yaml`."
        )
        raise FileNotFoundError(message)

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except (OSError, RuntimeError, ValueError) as error:
        message = f"Could not load the checkpoint: {checkpoint_path}. Error: {error}"
        raise RuntimeError(message) from error

    if not isinstance(checkpoint, dict):
        message = f"The checkpoint is not in the expected dictionary format: {checkpoint_path}"
        raise RuntimeError(message)

    required_keys = {"model_name", "model_state_dict", "label_mapping", "image_size"}
    missing_keys = required_keys.difference(checkpoint)
    if missing_keys:
        formatted_keys = ", ".join(sorted(missing_keys))
        message = f"The checkpoint is missing required fields: {formatted_keys}"
        raise RuntimeError(message)

    label_mapping = checkpoint["label_mapping"]
    if label_mapping != LABEL_TO_ID:
        message = (
            f"The checkpoint label mapping does not match the expected values. "
            f"Expected={LABEL_TO_ID}, received={label_mapping}"
        )
        raise RuntimeError(message)

    model_name = str(checkpoint["model_name"])
    model = build_model(
        model_name=model_name,
        num_classes=len(LABEL_TO_ID),
        pretrained=False,
        freeze_backbone=False,
    )
    try:
        model.load_state_dict(checkpoint["model_state_dict"])
    except RuntimeError as error:
        message = f"Checkpoint weights are incompatible with '{model_name}': {error}"
        raise RuntimeError(message) from error

    model.to(device)
    model.eval()
    return CheckpointBundle(
        model=model,
        metadata=checkpoint,
        checkpoint_path=checkpoint_path,
    )


def collect_test_predictions(
    *,
    model: nn.Module,
    dataloader: DataLoader[tuple[Tensor, Tensor]],
    device: torch.device,
    decision_threshold: float = DEFAULT_PNEUMONIA_THRESHOLD,
) -> PredictionBatch:
    """Run model inference on the test DataLoader.

    Args:
        model: Loaded classification model.
        dataloader: Test DataLoader.
        device: Runtime device.

    Returns:
        Ground-truth labels, predictions, and PNEUMONIA probabilities.
    """
    labels: list[int] = []
    predictions: list[int] = []
    probabilities: list[float] = []

    with torch.no_grad():
        for images, batch_labels in tqdm(dataloader, desc="Test evaluation", leave=False):
            images = images.to(device, non_blocking=True)
            logits = model(images)
            batch_probabilities = torch.softmax(logits, dim=1)
            batch_predictions = (
                batch_probabilities[:, POSITIVE_LABEL_ID] >= decision_threshold
            ).long()

            labels.extend(int(label) for label in batch_labels.cpu().tolist())
            predictions.extend(int(prediction) for prediction in batch_predictions.cpu().tolist())
            probabilities.extend(
                float(probability)
                for probability in batch_probabilities[:, POSITIVE_LABEL_ID].cpu().tolist()
            )

    if not labels:
        message = "The test DataLoader returned no samples; check the test manifest file."
        raise XRayDatasetError(message)

    return PredictionBatch(
        labels=labels,
        predictions=predictions,
        probabilities=probabilities,
    )


def calculate_metrics(prediction_batch: PredictionBatch) -> dict[str, Any]:
    """Calculate binary classification metrics for test predictions.

    Args:
        prediction_batch: Labels, predictions, and positive-class probabilities.

    Returns:
        JSON-serializable metrics payload.
    """
    labels = prediction_batch.labels
    predictions = prediction_batch.predictions
    probabilities = prediction_batch.probabilities
    matrix = confusion_matrix(
        labels,
        predictions,
        labels=[NEGATIVE_LABEL_ID, POSITIVE_LABEL_ID],
    )
    report = classification_report(
        labels,
        predictions,
        labels=[NEGATIVE_LABEL_ID, POSITIVE_LABEL_ID],
        target_names=CLASS_LABELS,
        output_dict=True,
        zero_division=0,
    )

    try:
        roc_auc = float(roc_auc_score(labels, probabilities))
    except ValueError:
        roc_auc = float("nan")

    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1_score": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": roc_auc,
        "confusion_matrix": {
            "labels": CLASS_LABELS,
            "matrix": matrix.astype(int).tolist(),
            "true_negative": int(matrix[0, 0]),
            "false_positive": int(matrix[0, 1]),
            "false_negative": int(matrix[1, 0]),
            "true_positive": int(matrix[1, 1]),
        },
        "classification_report": report,
        "sample_count": len(labels),
        "positive_label": ID_TO_LABEL[POSITIVE_LABEL_ID],
    }


def save_metrics_report(
    *,
    metrics_path: Path,
    metrics: dict[str, Any],
    checkpoint_bundle: CheckpointBundle,
    prediction_batch: PredictionBatch,
) -> None:
    """Save test metrics and checkpoint metadata as UTF-8 JSON.

    Args:
        metrics_path: Destination JSON path.
        metrics: Calculated metric payload.
        checkpoint_bundle: Loaded checkpoint and metadata.
        prediction_batch: Predictions used for the metrics.

    Raises:
        OSError: If the destination cannot be written.
    """
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint_path": str(checkpoint_bundle.checkpoint_path),
        "model_name": checkpoint_bundle.metadata["model_name"],
        "decision_threshold": checkpoint_bundle.metadata.get(
            "decision_threshold", DEFAULT_PNEUMONIA_THRESHOLD
        ),
        "image_size": checkpoint_bundle.metadata["image_size"],
        "normalization": checkpoint_bundle.metadata.get(
            "normalization",
            {
                "mean": IMAGENET_NORMALIZATION_MEAN,
                "std": IMAGENET_NORMALIZATION_STD,
            },
        ),
        "validation_metrics": checkpoint_bundle.metadata.get("validation_metrics", {}),
        "test_metrics": metrics,
        "predictions": [
            {
                "label_id": label,
                "label": ID_TO_LABEL[label],
                "prediction_id": prediction,
                "prediction": ID_TO_LABEL[prediction],
                "pneumonia_probability": probability,
            }
            for label, prediction, probability in zip(
                prediction_batch.labels,
                prediction_batch.predictions,
                prediction_batch.probabilities,
                strict=True,
            )
        ],
        "warning": MEDICAL_WARNING,
    }
    metrics_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=True),
        encoding="utf-8",
    )


def save_confusion_matrix_figure(figure_path: Path, metrics: dict[str, Any]) -> None:
    """Save a confusion matrix heatmap using matplotlib.

    Args:
        figure_path: Destination PNG path.
        metrics: Metrics payload containing confusion matrix values.

    Raises:
        OSError: If the figure cannot be written.
    """
    matrix = np.asarray(metrics["confusion_matrix"]["matrix"], dtype=np.int64)
    figure, axis = plt.subplots(figsize=(6, 5), constrained_layout=True)
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_title("Confusion Matrix")
    axis.set_xlabel("Predicted Label")
    axis.set_ylabel("True Label")
    axis.set_xticks(np.arange(len(CLASS_LABELS)), labels=CLASS_LABELS)
    axis.set_yticks(np.arange(len(CLASS_LABELS)), labels=CLASS_LABELS)

    threshold = matrix.max() / 2.0 if matrix.size else 0.0
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            text_color = "white" if value > threshold else "black"
            axis.text(
                column_index,
                row_index,
                str(value),
                ha="center",
                va="center",
                color=text_color,
                fontsize=12,
            )

    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_path, dpi=150)
    plt.close(figure)


def save_roc_curve_figure(
    figure_path: Path,
    prediction_batch: PredictionBatch,
    roc_auc: float,
) -> None:
    """Save ROC curve for the PNEUMONIA class.

    Args:
        figure_path: Destination PNG path.
        prediction_batch: Labels and positive-class probabilities.
        roc_auc: Area under ROC curve.

    Raises:
        OSError: If the figure cannot be written.
    """
    figure, axis = plt.subplots(figsize=(7, 5), constrained_layout=True)
    if len(set(prediction_batch.labels)) < 2:
        axis.text(
            0.5,
            0.5,
            "Could not generate an ROC curve: the test set contains only one class.",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
    else:
        false_positive_rate, true_positive_rate, _thresholds = roc_curve(
            prediction_batch.labels,
            prediction_batch.probabilities,
            pos_label=POSITIVE_LABEL_ID,
        )
        axis.plot(
            false_positive_rate,
            true_positive_rate,
            label=f"ROC-AUC = {roc_auc:.4f}",
            color="tab:blue",
            linewidth=2,
        )
        axis.plot([0, 1], [0, 1], linestyle="--", color="tab:gray", label="Random")
        axis.legend(loc="lower right")

    axis.set_title("ROC Curve")
    axis.set_xlabel("False Positive Rate")
    axis.set_ylabel("True Positive Rate")
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.05)
    axis.grid(alpha=0.3)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_path, dpi=150)
    plt.close(figure)


def save_precision_recall_curve_figure(
    figure_path: Path,
    prediction_batch: PredictionBatch,
) -> None:
    """Save precision-recall curve for the PNEUMONIA class.

    Args:
        figure_path: Destination PNG path.
        prediction_batch: Labels and positive-class probabilities.

    Raises:
        OSError: If the figure cannot be written.
    """
    figure, axis = plt.subplots(figsize=(7, 5), constrained_layout=True)
    if len(set(prediction_batch.labels)) < 2:
        axis.text(
            0.5,
            0.5,
            "Could not generate a precision-recall curve: the test set contains only one class.",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
    else:
        precision_values, recall_values, _thresholds = precision_recall_curve(
            prediction_batch.labels,
            prediction_batch.probabilities,
            pos_label=POSITIVE_LABEL_ID,
        )
        axis.plot(
            recall_values,
            precision_values,
            color="tab:green",
            linewidth=2,
            label="PNEUMONIA",
        )
        axis.legend(loc="lower left")

    axis.set_title("Precision-Recall Curve")
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.05)
    axis.grid(alpha=0.3)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_path, dpi=150)
    plt.close(figure)


def save_misclassified_examples_figure(
    *,
    figure_path: Path,
    dataset: ChestXRayDataset,
    prediction_batch: PredictionBatch,
    max_examples_per_group: int = 8,
) -> None:
    """Save a grid of false positive and false negative examples.

    Args:
        figure_path: Destination PNG path.
        dataset: Test dataset used for prediction metadata and image paths.
        prediction_batch: Labels, predictions, and probabilities.
        max_examples_per_group: Maximum number of false positives and false negatives shown.

    Raises:
        OSError: If image files or the destination cannot be read or written.
    """
    false_positive_indices = [
        index
        for index, (label, prediction) in enumerate(
            zip(prediction_batch.labels, prediction_batch.predictions, strict=True)
        )
        if label == NEGATIVE_LABEL_ID and prediction == POSITIVE_LABEL_ID
    ][:max_examples_per_group]
    false_negative_indices = [
        index
        for index, (label, prediction) in enumerate(
            zip(prediction_batch.labels, prediction_batch.predictions, strict=True)
        )
        if label == POSITIVE_LABEL_ID and prediction == NEGATIVE_LABEL_ID
    ][:max_examples_per_group]

    grouped_indices = [
        ("False Positive", false_positive_indices),
        ("False Negative", false_negative_indices),
    ]
    max_columns = max(max((len(indices) for _title, indices in grouped_indices), default=1), 1)
    max_columns = min(max_columns, max_examples_per_group)
    figure, axes = plt.subplots(
        nrows=2,
        ncols=max_columns,
        figsize=(max(4, max_columns * 2.4), 5.5),
        constrained_layout=True,
    )
    axes_array = np.asarray(axes).reshape(2, max_columns)

    for row_index, (group_title, sample_indices) in enumerate(grouped_indices):
        for column_index in range(max_columns):
            axis = axes_array[row_index, column_index]
            axis.axis("off")
            if column_index >= len(sample_indices):
                if column_index == 0 and not sample_indices:
                    axis.text(
                        0.5,
                        0.5,
                        f"{group_title}\nyok",
                        ha="center",
                        va="center",
                        transform=axis.transAxes,
                    )
                continue

            sample_index = sample_indices[column_index]
            sample = dataset.get_sample(sample_index)
            probability = prediction_batch.probabilities[sample_index]
            try:
                with Image.open(sample.filepath) as image:
                    axis.imshow(image.convert("L"), cmap="gray")
            except (OSError, UnidentifiedImageError) as error:
                message = (
                    f"Could not read the misclassified image: {sample.filepath}. Error: {error}"
                )
                raise OSError(message) from error

            axis.set_title(
                f"{group_title}\ntrue={sample.label}, p={probability:.2f}",
                fontsize=8,
            )

    figure.suptitle("Misclassified Examples", fontsize=13)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_path, dpi=150)
    plt.close(figure)


def save_evaluation_summary(
    *,
    summary_path: Path,
    metrics: dict[str, Any],
    checkpoint_bundle: CheckpointBundle,
    outputs: EvaluationOutputs,
) -> None:
    """Write a Markdown summary of test performance and clinical interpretation.

    Args:
        summary_path: Destination Markdown path.
        metrics: Calculated test metrics.
        checkpoint_bundle: Loaded checkpoint and metadata.
        outputs: Evaluation output file locations.

    Raises:
        OSError: If the summary cannot be written.
    """
    confusion = metrics["confusion_matrix"]
    strongest_metrics = identify_strongest_metrics(metrics)
    weakest_metrics = identify_weakest_metrics(metrics)
    false_positive_count = confusion["false_positive"]
    false_negative_count = confusion["false_negative"]

    false_positive_note = (
        f"- False positive count: `{false_positive_count}`. "
        "The model classified a NORMAL image as PNEUMONIA."
    )
    false_negative_note = (
        f"- False negative count: `{false_negative_count}`. "
        "The model classified a PNEUMONIA image as NORMAL."
    )
    recall_note = (
        "Recall measures the proportion of pneumonia-positive samples detected. "
        "Because PNEUMONIA is the positive class in this project, low recall means "
        "some images with pneumonia findings are missed and classified as NORMAL. "
        "This error can be more serious clinically; however, this project is intended "
        "only for educational and academic work."
    )

    decision_threshold = checkpoint_bundle.metadata.get(
        "decision_threshold", DEFAULT_PNEUMONIA_THRESHOLD
    )
    summary = f"""# Test Set Evaluation Summary

## Test Results

- Model: `{checkpoint_bundle.metadata["model_name"]}`
- Checkpoint: `{checkpoint_bundle.checkpoint_path}`
- Decision threshold: `{decision_threshold:.6f}`
- Test sample count: `{metrics["sample_count"]}`
- Accuracy: `{metrics["accuracy"]:.4f}`
- Precision: `{metrics["precision"]:.4f}`
- Recall: `{metrics["recall"]:.4f}`
- F1-score: `{metrics["f1_score"]:.4f}`
- ROC-AUC: `{format_metric_value(metrics["roc_auc"])}`

## Strongest Metrics

{strongest_metrics}

## Weaknesses

{weakest_metrics}

## False Positive / False Negative Interpretation

{false_positive_note}
{false_negative_note}
- Misclassified samples: `{outputs.misclassified_examples_path}`

## Why Does Recall Matter in a Medical Context?

{recall_note}

## Generated Outputs

- Metrics: `{outputs.metrics_path}`
- Confusion matrix: `{outputs.confusion_matrix_path}`
- ROC curve: `{outputs.roc_curve_path}`
- Precision-recall curve: `{outputs.precision_recall_curve_path}`
- Misclassified samples: `{outputs.misclassified_examples_path}`

## Warning

{MEDICAL_WARNING}
"""
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(summary, encoding="utf-8")


def identify_strongest_metrics(metrics: dict[str, Any]) -> str:
    """Return a Markdown list describing the strongest test metrics.

    Args:
        metrics: Calculated test metrics.

    Returns:
        Markdown bullet list.
    """
    metric_values = {
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1_score": metrics["f1_score"],
        "roc_auc": metrics["roc_auc"],
    }
    finite_metrics = {
        metric_name: metric_value
        for metric_name, metric_value in metric_values.items()
        if np.isfinite(metric_value)
    }
    if not finite_metrics:
        return "- No strongest metric could be identified; no computable metrics are available."

    sorted_metrics = sorted(finite_metrics.items(), key=lambda item: item[1], reverse=True)
    return "\n".join(
        f"- `{metric_name}`: `{metric_value:.4f}`"
        for metric_name, metric_value in sorted_metrics[:2]
    )


def identify_weakest_metrics(metrics: dict[str, Any]) -> str:
    """Return a Markdown list describing the weakest test metrics.

    Args:
        metrics: Calculated test metrics.

    Returns:
        Markdown bullet list.
    """
    metric_values = {
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1_score": metrics["f1_score"],
        "roc_auc": metrics["roc_auc"],
    }
    finite_metrics = {
        metric_name: metric_value
        for metric_name, metric_value in metric_values.items()
        if np.isfinite(metric_value)
    }
    if not finite_metrics:
        return "- No weakest metric could be identified; no computable metrics are available."

    sorted_metrics = sorted(finite_metrics.items(), key=lambda item: item[1])
    return "\n".join(
        f"- `{metric_name}`: `{metric_value:.4f}`"
        for metric_name, metric_value in sorted_metrics[:2]
    )


def format_metric_value(value: float) -> str:
    """Format a metric value for Markdown output.

    Args:
        value: Metric value that may be NaN.

    Returns:
        Four-decimal string or a descriptive missing value marker.
    """
    if not np.isfinite(value):
        return "not computable"
    return f"{value:.4f}"


def evaluate_model(config: ProjectConfig) -> EvaluationOutputs:
    """Run the complete test evaluation workflow.

    Args:
        config: Validated project configuration.

    Returns:
        Paths to generated evaluation outputs.

    Raises:
        FileNotFoundError: If the best model checkpoint is missing.
        RuntimeError: If model loading or evaluation fails.
        XRayDatasetError: If test data manifests are unavailable or invalid.
    """
    device = resolve_device(config.training.device)
    checkpoint_bundle = load_checkpoint_model(config.paths.best_model_path, device)
    test_dataset = ChestXRayDataset(
        config.paths.processed_data_dir / "test_manifest.csv",
        transform=CheckpointImageTransform.from_metadata(checkpoint_bundle.metadata),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        shuffle=False,
        pin_memory=config.data.pin_memory,
    )

    prediction_batch = collect_test_predictions(
        model=checkpoint_bundle.model,
        dataloader=test_loader,
        device=device,
        decision_threshold=float(
            checkpoint_bundle.metadata.get("decision_threshold", DEFAULT_PNEUMONIA_THRESHOLD)
        ),
    )
    metrics = calculate_metrics(prediction_batch)

    outputs = EvaluationOutputs(
        metrics_path=config.paths.metrics_dir / "test_metrics.json",
        summary_path=Path("docs/evaluation_summary.md"),
        confusion_matrix_path=config.paths.figures_dir / "confusion_matrix.png",
        roc_curve_path=config.paths.figures_dir / "roc_curve.png",
        precision_recall_curve_path=config.paths.figures_dir / "precision_recall_curve.png",
        misclassified_examples_path=config.paths.figures_dir / "misclassified_examples.png",
    )

    save_metrics_report(
        metrics_path=outputs.metrics_path,
        metrics=metrics,
        checkpoint_bundle=checkpoint_bundle,
        prediction_batch=prediction_batch,
    )
    save_confusion_matrix_figure(outputs.confusion_matrix_path, metrics)
    save_roc_curve_figure(
        outputs.roc_curve_path,
        prediction_batch,
        float(metrics["roc_auc"]),
    )
    save_precision_recall_curve_figure(outputs.precision_recall_curve_path, prediction_batch)
    save_misclassified_examples_figure(
        figure_path=outputs.misclassified_examples_path,
        dataset=test_dataset,
        prediction_batch=prediction_batch,
    )
    save_evaluation_summary(
        summary_path=outputs.summary_path,
        metrics=metrics,
        checkpoint_bundle=checkpoint_bundle,
        outputs=outputs,
    )
    return outputs


def parse_args() -> argparse.Namespace:
    """Parse evaluation command line arguments.

    Returns:
        Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate the best trained model on the test set."
    )
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to the YAML configuration file.",
    )
    return parser.parse_args()


def main() -> None:
    """Run test-set evaluation from the command line.

    Raises:
        ConfigFileError: If the config file is invalid.
        FileNotFoundError: If the checkpoint is missing.
        RuntimeError: If checkpoint loading or evaluation fails.
    """
    args = parse_args()
    try:
        config = load_and_prepare_config(args.config)
    except ConfigFileError:
        raise

    outputs = evaluate_model(config)
    metrics_payload = json.loads(outputs.metrics_path.read_text(encoding="utf-8"))
    test_metrics = metrics_payload["test_metrics"]
    print("Test evaluation completed.")
    print(f"accuracy={test_metrics['accuracy']:.4f}")
    print(f"precision={test_metrics['precision']:.4f}")
    print(f"recall={test_metrics['recall']:.4f}")
    print(f"f1_score={test_metrics['f1_score']:.4f}")
    print(f"roc_auc={format_metric_value(float(test_metrics['roc_auc']))}")
    print(f"Metrics: {outputs.metrics_path}")
    print(f"Summary: {outputs.summary_path}")
    print(f"Confusion matrix: {outputs.confusion_matrix_path}")
    print(f"ROC curve: {outputs.roc_curve_path}")
    print(f"Precision-recall curve: {outputs.precision_recall_curve_path}")
    print(f"Misclassified examples: {outputs.misclassified_examples_path}")


if __name__ == "__main__":
    main()
