"""DataLoader and class-imbalance utilities for chest X-ray manifests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.core.config import ProjectConfig
from src.data.dataset import ChestXRayDataset, LABEL_TO_ID, XRayDatasetError
from src.data.transforms import build_eval_transforms, build_train_transforms


EXPECTED_SPLITS: tuple[str, str, str] = ("train", "val", "test")


@dataclass(frozen=True)
class XRayDataLoaders:
    """Train, validation, and test DataLoader bundle.

    Attributes:
        train: DataLoader for training samples.
        val: DataLoader for validation samples.
        test: DataLoader for test samples.
        class_weights: Tensor of inverse-frequency class weights ordered as NORMAL, PNEUMONIA.
    """

    train: DataLoader[tuple[Tensor, Tensor]]
    val: DataLoader[tuple[Tensor, Tensor]]
    test: DataLoader[tuple[Tensor, Tensor]]
    class_weights: Tensor


def create_dataloaders(config: ProjectConfig) -> XRayDataLoaders:
    """Create train, validation, and test DataLoaders from configured manifests.

    Args:
        config: Loaded project configuration.

    Returns:
        DataLoader bundle and training class weights.

    Raises:
        XRayDatasetError: If a required manifest is missing or invalid.
        ValueError: If class weights cannot be computed from the training dataset.
    """
    manifest_paths = get_manifest_paths(config.paths.processed_data_dir)
    train_dataset = ChestXRayDataset(
        manifest_paths["train"],
        transform=build_train_transforms(config.data.image_size),
    )
    val_dataset = ChestXRayDataset(
        manifest_paths["val"],
        transform=build_eval_transforms(config.data.image_size),
    )
    test_dataset = ChestXRayDataset(
        manifest_paths["test"],
        transform=build_eval_transforms(config.data.image_size),
    )
    class_weights = calculate_class_weights(train_dataset)

    sampler = (
        build_weighted_sampler(train_dataset)
        if config.training.use_weighted_sampler
        else None
    )
    train_shuffle = sampler is None

    return XRayDataLoaders(
        train=DataLoader(
            train_dataset,
            batch_size=config.data.batch_size,
            shuffle=train_shuffle,
            sampler=sampler,
            num_workers=config.data.num_workers,
            pin_memory=config.data.pin_memory,
        ),
        val=DataLoader(
            val_dataset,
            batch_size=config.data.batch_size,
            shuffle=False,
            num_workers=config.data.num_workers,
            pin_memory=config.data.pin_memory,
        ),
        test=DataLoader(
            test_dataset,
            batch_size=config.data.batch_size,
            shuffle=False,
            num_workers=config.data.num_workers,
            pin_memory=config.data.pin_memory,
        ),
        class_weights=class_weights,
    )


def get_manifest_paths(processed_data_dir: Path) -> dict[str, Path]:
    """Build expected split manifest paths.

    Args:
        processed_data_dir: Directory containing standardized manifest CSV files.

    Returns:
        Split names mapped to manifest file paths.
    """
    return {split: processed_data_dir / f"{split}_manifest.csv" for split in EXPECTED_SPLITS}


def calculate_class_weights(dataset: ChestXRayDataset) -> Tensor:
    """Calculate inverse-frequency class weights for weighted cross entropy.

    Args:
        dataset: Training dataset with manifest samples.

    Returns:
        Float tensor ordered by numeric class id.

    Raises:
        ValueError: If a class has zero samples.
    """
    class_counts = count_labels(dataset)
    total_count = sum(class_counts.values())
    class_count = len(LABEL_TO_ID)
    weights: list[float] = []

    for label_id in sorted(class_counts):
        sample_count = class_counts[label_id]
        if sample_count <= 0:
            message = f"Class weight hesaplanamadı; label_id={label_id} için örnek yok."
            raise ValueError(message)
        weights.append(total_count / (class_count * sample_count))

    return torch.tensor(weights, dtype=torch.float32)


def build_weighted_sampler(dataset: ChestXRayDataset) -> WeightedRandomSampler:
    """Create a WeightedRandomSampler using inverse class frequencies.

    Args:
        dataset: Training dataset with manifest samples.

    Returns:
        Sampler that draws samples with replacement using per-sample weights.

    Raises:
        ValueError: If a class has zero samples.
    """
    class_counts = count_labels(dataset)
    sample_weights = [
        1.0 / class_counts[sample.label_id]
        for sample in dataset.samples
    ]
    return WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )


def count_labels(dataset: ChestXRayDataset) -> dict[int, int]:
    """Count labels in a manifest-backed dataset.

    Args:
        dataset: Dataset whose `samples` metadata should be counted.

    Returns:
        Numeric label ids mapped to sample counts.
    """
    class_counts = {label_id: 0 for label_id in LABEL_TO_ID.values()}
    for sample in dataset.samples:
        class_counts[sample.label_id] += 1
    return class_counts


def require_manifest_files(processed_data_dir: Path) -> None:
    """Validate that all split manifest files exist before training.

    Args:
        processed_data_dir: Directory containing standardized manifest CSV files.

    Raises:
        XRayDatasetError: If one or more manifest files are missing.
    """
    missing_paths = [
        path
        for path in get_manifest_paths(processed_data_dir).values()
        if not path.is_file()
    ]
    if missing_paths:
        formatted_paths = "\n".join(f"- {path}" for path in missing_paths)
        message = (
            "Manifest dosyaları bulunamadı. Önce şu komutu çalıştırın: "
            "`python -m src.data.standardize_dataset --config configs/config.yaml`\n"
            f"Eksik dosyalar:\n{formatted_paths}"
        )
        raise XRayDatasetError(message)
