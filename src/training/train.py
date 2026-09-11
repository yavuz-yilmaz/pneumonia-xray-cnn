"""Training entry point for baseline and transfer-learning chest X-ray models."""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.core.config import ConfigFileError, ProjectConfig, load_and_prepare_config
from src.core.reproducibility import set_global_seed
from src.data.dataloader import create_dataloaders, require_manifest_files
from src.data.dataset import ID_TO_LABEL, LABEL_TO_ID
from src.data.transforms import IMAGENET_NORMALIZATION_MEAN, IMAGENET_NORMALIZATION_STD
from src.training.models import build_model, count_trainable_parameters, unfreeze_fine_tuning_layers


@dataclass(frozen=True)
class EpochMetrics:
    """Metrics produced for one training epoch.

    Attributes:
        train_loss: Mean training loss.
        validation_loss: Mean validation loss.
        validation_accuracy: Validation accuracy.
        validation_precision: Validation precision for the PNEUMONIA class.
        validation_recall: Validation recall for the PNEUMONIA class.
        validation_f1: Validation F1-score for the PNEUMONIA class.
        learning_rate: Optimizer learning rate at the end of the epoch.
    """

    train_loss: float
    validation_loss: float
    validation_accuracy: float
    validation_precision: float
    validation_recall: float
    validation_f1: float
    learning_rate: float
    phase: str


@dataclass(frozen=True)
class TrainingResult:
    """Final result metadata from a training run.

    Attributes:
        best_epoch: One-based epoch index with the best validation loss.
        best_validation_loss: Best validation loss observed.
        best_metrics: Full metric set for the best epoch.
        checkpoint_path: Path where the best checkpoint was saved.
        history_path: Path where JSON history was saved.
        figure_path: Path where training curves were saved.
    """

    best_epoch: int
    best_validation_loss: float
    best_metrics: EpochMetrics
    checkpoint_path: Path
    history_path: Path
    figure_path: Path


def resolve_device(device_name: str) -> torch.device:
    """Resolve configured device name to an available PyTorch device.

    Args:
        device_name: One of `auto`, `cpu`, `cuda`, or `mps`.

    Returns:
        A concrete PyTorch device.

    Raises:
        RuntimeError: If a requested accelerator is unavailable.
    """
    normalized_device_name = device_name.lower()
    if normalized_device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if normalized_device_name == "cuda" and not torch.cuda.is_available():
        message = "Config device='cuda' seçilmiş ancak CUDA kullanılabilir değil."
        raise RuntimeError(message)
    if normalized_device_name == "mps" and not torch.backends.mps.is_available():
        message = "Config device='mps' seçilmiş ancak Apple MPS kullanılabilir değil."
        raise RuntimeError(message)
    return torch.device(normalized_device_name)


def train_model(
    config: ProjectConfig,
    *,
    model_name: str | None = None,
    checkpoint_filename: str | None = None,
    history_filename: str | None = None,
    figure_filename: str | None = None,
    early_stopping_patience: int | None = None,
) -> TrainingResult:
    """Train a model and persist the best checkpoint, history, and training curves.

    Args:
        config: Validated project configuration.
        model_name: Architecture name to train. Uses `config.model.name` when omitted.
        checkpoint_filename: Filename under `config.paths.model_dir` for the best checkpoint.
        history_filename: Filename under `config.paths.metrics_dir` for JSON history.
        figure_filename: Filename under `config.paths.figures_dir` for curves.
        early_stopping_patience: Stop after this many non-improving epochs per training phase.

    Returns:
        Result metadata for the training run.

    Raises:
        ValueError: If early stopping patience is invalid.
        RuntimeError: If training cannot produce a best model checkpoint.
    """
    resolved_model_name = config.model.name if model_name is None else model_name
    resolved_checkpoint_filename = (
        "best_model.pt" if checkpoint_filename is None else checkpoint_filename
    )
    resolved_history_filename = (
        "transfer_learning_history.json" if history_filename is None else history_filename
    )
    resolved_figure_filename = (
        "transfer_learning_training_curves.png" if figure_filename is None else figure_filename
    )
    resolved_early_stopping_patience = (
        config.training.early_stopping_patience
        if early_stopping_patience is None
        else early_stopping_patience
    )
    if resolved_early_stopping_patience < 1:
        message = (
            "early_stopping_patience pozitif bir tam sayı olmalı; "
            f"alınan değer: {resolved_early_stopping_patience}"
        )
        raise ValueError(message)

    set_global_seed(config.training.seed)
    require_manifest_files(config.paths.processed_data_dir)

    device = resolve_device(config.training.device)
    dataloaders = create_dataloaders(config)
    model = build_model(
        model_name=resolved_model_name,
        num_classes=config.model.num_classes,
        pretrained=config.model.pretrained,
        freeze_backbone=config.model.freeze_backbone,
    ).to(device)
    class_weights = (
        dataloaders.class_weights.to(device) if config.training.use_weighted_loss else None
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    best_validation_loss = float("inf")
    best_epoch = 0
    best_model_state: dict[str, Tensor] | None = None
    best_metrics: EpochMetrics | None = None
    history: list[dict[str, float | int | str]] = []
    start_time = perf_counter()

    print(
        "Eğitim başlıyor: "
        f"model={resolved_model_name}, device={device}, epochs={config.training.epochs}, "
        f"fine_tune_epochs={config.training.fine_tune_epochs}, "
        f"trainable_parameters={count_trainable_parameters(model)}"
    )

    training_phases = resolve_phase_learning_rates(
        build_training_phases(
            model_name=resolved_model_name,
            freeze_backbone=config.model.freeze_backbone,
            classifier_epochs=config.training.epochs,
            fine_tune_epochs=config.training.fine_tune_epochs,
        ),
        classifier_learning_rate=config.training.learning_rate,
        fine_tune_learning_rate=config.training.fine_tune_learning_rate,
    )
    global_epoch = 0

    for phase_name, phase_epochs, phase_learning_rate in training_phases:
        if phase_name == "fine_tuning":
            unfreeze_fine_tuning_layers(model, resolved_model_name)
            print(
                "Fine-tuning aşaması başlıyor: "
                f"trainable_parameters={count_trainable_parameters(model)}, "
                f"learning_rate={phase_learning_rate}"
            )

        use_mixed_precision = device.type == "cuda"
        gradient_scaler = torch.amp.GradScaler(device.type, enabled=use_mixed_precision)
        optimizer = AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=phase_learning_rate,
            weight_decay=config.training.weight_decay,
        )
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=1,
            min_lr=1e-7,
        )
        epochs_without_improvement = 0

        for phase_epoch_index in range(phase_epochs):
            global_epoch += 1
            train_loss = run_training_epoch(
                model=model,
                dataloader=dataloaders.train,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                epoch_index=phase_epoch_index,
                epoch_count=phase_epochs,
                use_mixed_precision=use_mixed_precision,
                gradient_scaler=gradient_scaler,
            )
            validation_metrics = evaluate_validation_epoch(
                model=model,
                dataloader=dataloaders.val,
                criterion=criterion,
                device=device,
            )
            current_learning_rate = optimizer.param_groups[0]["lr"]
            epoch_metrics = EpochMetrics(
                train_loss=train_loss,
                validation_loss=validation_metrics["loss"],
                validation_accuracy=validation_metrics["accuracy"],
                validation_precision=validation_metrics["precision"],
                validation_recall=validation_metrics["recall"],
                validation_f1=validation_metrics["f1"],
                learning_rate=float(current_learning_rate),
                phase=phase_name,
            )
            scheduler.step(epoch_metrics.validation_loss)

            history_entry = {
                "epoch": global_epoch,
                "phase_epoch": phase_epoch_index + 1,
                "phase": phase_name,
                "train_loss": epoch_metrics.train_loss,
                "validation_loss": epoch_metrics.validation_loss,
                "validation_accuracy": epoch_metrics.validation_accuracy,
                "validation_precision": epoch_metrics.validation_precision,
                "validation_recall": epoch_metrics.validation_recall,
                "validation_f1": epoch_metrics.validation_f1,
                "learning_rate": epoch_metrics.learning_rate,
            }
            history.append(history_entry)
            print(format_epoch_metrics(history_entry))

            if epoch_metrics.validation_loss < best_validation_loss:
                best_validation_loss = epoch_metrics.validation_loss
                best_epoch = global_epoch
                best_metrics = epoch_metrics
                best_model_state = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= resolved_early_stopping_patience:
                    print(
                        "Early stopping tetiklendi: "
                        f"{resolved_early_stopping_patience} epoch boyunca "
                        f"{phase_name} validation loss iyileşmedi."
                    )
                    break

    if best_model_state is None or best_metrics is None:
        message = "Eğitim tamamlandı ancak en iyi model durumu kaydedilemedi."
        raise RuntimeError(message)

    elapsed_seconds = perf_counter() - start_time
    checkpoint_path = config.paths.model_dir / resolved_checkpoint_filename
    history_path = config.paths.metrics_dir / resolved_history_filename
    figure_path = config.paths.figures_dir / resolved_figure_filename

    save_checkpoint(
        checkpoint_path=checkpoint_path,
        model_name=resolved_model_name,
        model_state=best_model_state,
        config=config,
        best_epoch=best_epoch,
        best_metrics=best_metrics,
    )
    save_history(
        history_path=history_path,
        history=history,
        model_name=resolved_model_name,
        checkpoint_path=checkpoint_path,
        elapsed_seconds=elapsed_seconds,
    )
    save_training_curves(
        figure_path=figure_path,
        history=history,
        model_name=resolved_model_name,
    )

    return TrainingResult(
        best_epoch=best_epoch,
        best_validation_loss=best_validation_loss,
        best_metrics=best_metrics,
        checkpoint_path=checkpoint_path,
        history_path=history_path,
        figure_path=figure_path,
    )


def build_training_phases(
    *,
    model_name: str,
    freeze_backbone: bool,
    classifier_epochs: int,
    fine_tune_epochs: int,
) -> list[tuple[str, int, float]]:
    """Create training phase descriptors for baseline or transfer-learning runs.

    Args:
        model_name: Selected architecture name.
        freeze_backbone: Whether the initial transfer-learning phase freezes the backbone.
        classifier_epochs: Number of epochs for the initial phase.
        fine_tune_epochs: Number of epochs for the fine-tuning phase.

    Returns:
        List of `(phase_name, epoch_count, learning_rate)` tuples. The learning-rate
        placeholder is filled by `train_model` to keep the phase order testable.
    """
    if classifier_epochs < 1:
        message = f"classifier_epochs pozitif olmalı; alınan değer: {classifier_epochs}"
        raise ValueError(message)
    if fine_tune_epochs < 0:
        message = f"fine_tune_epochs negatif olamaz; alınan değer: {fine_tune_epochs}"
        raise ValueError(message)

    normalized_model_name = model_name.strip().lower()
    if normalized_model_name == "simple_cnn":
        return [("baseline", classifier_epochs, 0.0)]

    phases = [("classifier", classifier_epochs, 0.0)]
    if freeze_backbone and fine_tune_epochs > 0:
        phases.append(("fine_tuning", fine_tune_epochs, 0.0))
    return phases


def resolve_phase_learning_rates(
    phases: list[tuple[str, int, float]],
    *,
    classifier_learning_rate: float,
    fine_tune_learning_rate: float,
) -> list[tuple[str, int, float]]:
    """Attach learning rates to training phase descriptors.

    Args:
        phases: Phase descriptors from `build_training_phases`.
        classifier_learning_rate: Learning rate for baseline or classifier-only training.
        fine_tune_learning_rate: Lower learning rate for fine-tuning.

    Returns:
        Phase descriptors with concrete learning rates.
    """
    resolved_phases: list[tuple[str, int, float]] = []
    for phase_name, phase_epochs, _placeholder_learning_rate in phases:
        learning_rate = (
            fine_tune_learning_rate if phase_name == "fine_tuning" else classifier_learning_rate
        )
        resolved_phases.append((phase_name, phase_epochs, learning_rate))
    return resolved_phases


def run_training_epoch(
    *,
    model: nn.Module,
    dataloader: DataLoader[tuple[Tensor, Tensor]],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch_index: int,
    epoch_count: int,
    use_mixed_precision: bool,
    gradient_scaler: torch.amp.GradScaler,
) -> float:
    """Train a model for one epoch.

    Args:
        model: Model to update.
        dataloader: Training DataLoader.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Runtime device.
        epoch_index: Zero-based epoch index.
        epoch_count: Total configured epochs.
        use_mixed_precision: Whether to run forward/loss computation in mixed precision.
        gradient_scaler: CUDA gradient scaler used when mixed precision is enabled.

    Returns:
        Mean loss across the epoch.
    """
    model.train()
    running_loss = 0.0
    sample_count = 0
    progress = tqdm(
        dataloader,
        desc=f"Epoch {epoch_index + 1}/{epoch_count} train",
        leave=False,
    )

    for images, labels in progress:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_mixed_precision):
            logits = model(images)
            loss = criterion(logits, labels)

        gradient_scaler.scale(loss).backward()
        gradient_scaler.step(optimizer)
        gradient_scaler.update()

        batch_size = images.size(0)
        running_loss += float(loss.item()) * batch_size
        sample_count += batch_size
        progress.set_postfix(loss=running_loss / max(sample_count, 1))

    return running_loss / max(sample_count, 1)


def evaluate_validation_epoch(
    *,
    model: nn.Module,
    dataloader: DataLoader[tuple[Tensor, Tensor]],
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate validation loss and binary classification metrics.

    Args:
        model: Model to evaluate.
        dataloader: Validation DataLoader.
        criterion: Loss function.
        device: Runtime device.

    Returns:
        Validation loss, accuracy, precision, recall, and F1-score.
    """
    model.eval()
    running_loss = 0.0
    sample_count = 0
    all_predictions: list[int] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validation", leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, labels)
            predictions = torch.argmax(logits, dim=1)

            batch_size = images.size(0)
            running_loss += float(loss.item()) * batch_size
            sample_count += batch_size
            all_predictions.extend(predictions.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    return {
        "loss": running_loss / max(sample_count, 1),
        "accuracy": accuracy_score(all_labels, all_predictions),
        "precision": precision_score(all_labels, all_predictions, zero_division=0),
        "recall": recall_score(all_labels, all_predictions, zero_division=0),
        "f1": f1_score(all_labels, all_predictions, zero_division=0),
    }


def save_checkpoint(
    *,
    checkpoint_path: Path,
    model_name: str,
    model_state: dict[str, Tensor],
    config: ProjectConfig,
    best_epoch: int,
    best_metrics: EpochMetrics,
) -> None:
    """Save the best model checkpoint with inference metadata.

    Args:
        checkpoint_path: Destination checkpoint path.
        model_name: Architecture name.
        model_state: Best model state dict.
        config: Project configuration used for training.
        best_epoch: One-based epoch index for the best model.
        best_metrics: Validation metrics for the best model.

    Raises:
        OSError: If the destination directory cannot be written.
    """
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_name": model_name,
        "model_state_dict": model_state,
        "label_mapping": LABEL_TO_ID,
        "id_to_label": ID_TO_LABEL,
        "image_size": config.data.image_size,
        "normalization": {
            "mean": IMAGENET_NORMALIZATION_MEAN,
            "std": IMAGENET_NORMALIZATION_STD,
        },
        "best_epoch": best_epoch,
        "validation_metrics": {
            "loss": best_metrics.validation_loss,
            "accuracy": best_metrics.validation_accuracy,
            "precision": best_metrics.validation_precision,
            "recall": best_metrics.validation_recall,
            "f1": best_metrics.validation_f1,
        },
        "config": config.model_dump(mode="json"),
    }
    torch.save(checkpoint, checkpoint_path)


def save_history(
    *,
    history_path: Path,
    history: list[dict[str, float | int | str]],
    model_name: str,
    checkpoint_path: Path,
    elapsed_seconds: float,
) -> None:
    """Save training history as UTF-8 JSON.

    Args:
        history_path: Destination JSON path.
        history: Per-epoch metrics.
        model_name: Trained architecture name.
        checkpoint_path: Best checkpoint path.
        elapsed_seconds: Training runtime in seconds.

    Raises:
        OSError: If the destination cannot be written.
    """
    history_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model_name": model_name,
        "checkpoint_path": str(checkpoint_path),
        "elapsed_seconds": elapsed_seconds,
        "history": history,
    }
    history_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def save_training_curves(
    *,
    figure_path: Path,
    history: list[dict[str, float | int | str]],
    model_name: str,
) -> None:
    """Save loss and validation accuracy curves.

    Args:
        figure_path: Destination PNG path.
        history: Per-epoch metric dictionaries.
        model_name: Model name used in the figure title.

    Raises:
        OSError: If the figure cannot be written.
    """
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [int(entry["epoch"]) for entry in history]
    train_losses = [float(entry["train_loss"]) for entry in history]
    validation_losses = [float(entry["validation_loss"]) for entry in history]
    validation_accuracies = [float(entry["validation_accuracy"]) for entry in history]

    figure, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5), constrained_layout=True)
    axes[0].plot(epochs, train_losses, marker="o", label="Train loss")
    axes[0].plot(epochs, validation_losses, marker="o", label="Validation loss")
    axes[0].set_title(f"{model_name} Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, validation_accuracies, marker="o", color="tab:green")
    axes[1].set_title("Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(alpha=0.3)

    figure.savefig(figure_path, dpi=150)
    plt.close(figure)


def format_epoch_metrics(history_entry: dict[str, float | int | str]) -> str:
    """Format epoch metrics for console output.

    Args:
        history_entry: One serialized epoch history entry.

    Returns:
        Human-readable one-line metric summary.
    """
    return (
        f"Epoch {history_entry['epoch']}: "
        f"phase={history_entry['phase']}, "
        f"train_loss={float(history_entry['train_loss']):.4f}, "
        f"val_loss={float(history_entry['validation_loss']):.4f}, "
        f"val_acc={float(history_entry['validation_accuracy']):.4f}, "
        f"val_precision={float(history_entry['validation_precision']):.4f}, "
        f"val_recall={float(history_entry['validation_recall']):.4f}, "
        f"val_f1={float(history_entry['validation_f1']):.4f}"
    )


def parse_args() -> argparse.Namespace:
    """Parse training command line arguments.

    Returns:
        Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(description="Baseline veya transfer learning modelini eğit.")
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="YAML config dosyası yolu.",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Eğitilecek model adı. Verilmezse config içindeki model.name kullanılır.",
    )
    return parser.parse_args()


def main() -> None:
    """Run model training from the command line.

    Raises:
        ConfigFileError: If the config file is invalid.
        RuntimeError: If training fails to produce outputs.
    """
    args = parse_args()
    try:
        config = load_and_prepare_config(args.config)
    except ConfigFileError:
        raise

    result = train_model(config, model_name=args.model_name)
    print(
        "Eğitim tamamlandı. "
        f"best_epoch={result.best_epoch}, "
        f"val_loss={result.best_metrics.validation_loss:.4f}, "
        f"val_acc={result.best_metrics.validation_accuracy:.4f}, "
        f"val_precision={result.best_metrics.validation_precision:.4f}, "
        f"val_recall={result.best_metrics.validation_recall:.4f}, "
        f"val_f1={result.best_metrics.validation_f1:.4f}"
    )
    print(f"Checkpoint: {result.checkpoint_path}")
    print(f"History: {result.history_path}")
    print(f"Figure: {result.figure_path}")


if __name__ == "__main__":
    main()
