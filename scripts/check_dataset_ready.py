"""Check that the manually downloaded chest X-ray dataset is ready to use."""

from __future__ import annotations

import sys
from pathlib import Path

DATA_ROOT = Path("data/raw/chest_xray")
EXPECTED_SPLITS = ("train", "val", "test")
EXPECTED_CLASSES = ("NORMAL", "PNEUMONIA")
SUPPORTED_EXTENSIONS = {".jpeg", ".jpg", ".png"}


class DatasetReadinessError(RuntimeError):
    """Raised when the manually placed dataset is missing or malformed."""


def find_supported_images(directory: Path) -> list[Path]:
    """Return supported image files directly under a class directory.

    Args:
        directory: Class directory to scan.

    Returns:
        Sorted image paths with supported extensions.
    """
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def validate_dataset_structure(data_root: Path = DATA_ROOT) -> dict[str, dict[str, int]]:
    """Validate the expected dataset folder structure and count images.

    Args:
        data_root: Root directory containing train, val, and test folders.

    Returns:
        Nested mapping of split and class names to image counts.

    Raises:
        DatasetReadinessError: If required folders are missing or any class is empty.
    """
    if not data_root.exists():
        message = (
            f"Dataset directory not found: {data_root}\n"
            "Download and extract the Chest X-Ray Images (Pneumonia) dataset manually "
            "and place the `chest_xray` directory at `data/raw/chest_xray`."
        )
        raise DatasetReadinessError(message)
    if not data_root.is_dir():
        message = f"The dataset path must be a directory, but a file was found: {data_root}"
        raise DatasetReadinessError(message)

    summary: dict[str, dict[str, int]] = {}
    missing_paths: list[Path] = []
    empty_classes: list[Path] = []

    for split in EXPECTED_SPLITS:
        split_dir = data_root / split
        if not split_dir.is_dir():
            missing_paths.append(split_dir)
            continue

        summary[split] = {}
        for class_name in EXPECTED_CLASSES:
            class_dir = split_dir / class_name
            if not class_dir.is_dir():
                missing_paths.append(class_dir)
                continue

            image_count = len(find_supported_images(class_dir))
            summary[split][class_name] = image_count
            if image_count == 0:
                empty_classes.append(class_dir)

    if missing_paths:
        formatted_paths = "\n".join(f"- {path}" for path in missing_paths)
        message = (
            "The dataset directory structure is incomplete. Missing directories:\n"
            f"{formatted_paths}\n\n"
            "Expected structure: data/raw/chest_xray/{train,val,test}/{NORMAL,PNEUMONIA}"
        )
        raise DatasetReadinessError(message)

    if empty_classes:
        formatted_paths = "\n".join(f"- {path}" for path in empty_classes)
        supported_extensions = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        message = (
            "Some class directories contain no supported images:\n"
            f"{formatted_paths}\n\n"
            f"Supported extensions: {supported_extensions}"
        )
        raise DatasetReadinessError(message)

    return summary


def print_summary(summary: dict[str, dict[str, int]]) -> None:
    """Print a compact dataset readiness summary.

    Args:
        summary: Nested mapping of split and class names to image counts.
    """
    print("The dataset directory structure is ready.")
    print(f"Root directory: {DATA_ROOT}")
    for split in EXPECTED_SPLITS:
        normal_count = summary[split]["NORMAL"]
        pneumonia_count = summary[split]["PNEUMONIA"]
        total_count = normal_count + pneumonia_count
        print(
            f"- {split}: NORMAL={normal_count}, PNEUMONIA={pneumonia_count}, total={total_count}"
        )


def main() -> int:
    """Run the dataset readiness check.

    Returns:
        Process exit code.
    """
    try:
        summary = validate_dataset_structure(DATA_ROOT)
    except DatasetReadinessError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
