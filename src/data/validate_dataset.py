"""Validate image readability and summarize the chest X-ray dataset."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

from src.core.config import ConfigFileError, ProjectConfig, load_and_prepare_config


EXPECTED_SPLITS = ("train", "val", "test")
EXPECTED_CLASSES = ("NORMAL", "PNEUMONIA")
SUPPORTED_EXTENSIONS = {".jpeg", ".jpg", ".png"}


class DatasetValidationError(RuntimeError):
    """Raised when the dataset cannot be validated because required inputs are missing."""


@dataclass(frozen=True)
class ImageValidationResult:
    """Readability result for one image file."""

    filepath: Path
    split: str
    label: str
    is_valid: bool
    error: str | None


def collect_image_paths(data_root: Path) -> dict[str, dict[str, list[Path]]]:
    """Collect supported image paths from the expected dataset structure.

    Args:
        data_root: Dataset root directory.

    Returns:
        Nested mapping of split and label to sorted image paths.

    Raises:
        DatasetValidationError: If required directories are missing.
    """
    if not data_root.is_dir():
        message = (
            f"Dataset directory not found: {data_root}. "
            "Place the `chest_xray` directory at `data/raw/chest_xray`."
        )
        raise DatasetValidationError(message)

    image_paths: dict[str, dict[str, list[Path]]] = {}
    missing_directories: list[Path] = []
    for split in EXPECTED_SPLITS:
        image_paths[split] = {}
        for label in EXPECTED_CLASSES:
            class_dir = data_root / split / label
            if not class_dir.is_dir():
                missing_directories.append(class_dir)
                image_paths[split][label] = []
                continue
            image_paths[split][label] = sorted(
                path
                for path in class_dir.iterdir()
                if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
            )

    if missing_directories:
        formatted_paths = "\n".join(f"- {path}" for path in missing_directories)
        message = f"The dataset directory structure is incomplete:\n{formatted_paths}"
        raise DatasetValidationError(message)

    return image_paths


def validate_image_file(filepath: Path, split: str, label: str) -> ImageValidationResult:
    """Check whether one image can be opened and decoded by Pillow.

    Args:
        filepath: Image path to validate.
        split: Dataset split name.
        label: Class label name.

    Returns:
        Image validation result.
    """
    try:
        with Image.open(filepath) as image:
            image.verify()
        with Image.open(filepath) as image:
            image.load()
    except (OSError, UnidentifiedImageError) as error:
        return ImageValidationResult(
            filepath=filepath,
            split=split,
            label=label,
            is_valid=False,
            error=str(error),
        )
    return ImageValidationResult(
        filepath=filepath,
        split=split,
        label=label,
        is_valid=True,
        error=None,
    )


def build_dataset_summary(
    image_paths: dict[str, dict[str, list[Path]]],
    corrupted_images: list[dict[str, str]],
) -> dict[str, Any]:
    """Build serializable split and class count summary.

    Args:
        image_paths: Nested mapping of split and label to image paths.
        corrupted_images: Corrupted image records.

    Returns:
        JSON-serializable dataset summary.
    """
    split_summary: dict[str, dict[str, int]] = {}
    total_by_class = dict.fromkeys(EXPECTED_CLASSES, 0)
    total_images = 0

    for split in EXPECTED_SPLITS:
        split_summary[split] = {}
        for label in EXPECTED_CLASSES:
            image_count = len(image_paths[split][label])
            split_summary[split][label] = image_count
            total_by_class[label] += image_count
            total_images += image_count
        split_summary[split]["total"] = sum(split_summary[split].values())

    return {
        "data_root": str(Path("data/raw/chest_xray")),
        "supported_extensions": sorted(SUPPORTED_EXTENSIONS),
        "splits": split_summary,
        "total_by_class": total_by_class,
        "total_images": total_images,
        "corrupted_image_count": len(corrupted_images),
    }


def validate_dataset(config: ProjectConfig) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Validate configured dataset images and write no files.

    Args:
        config: Loaded project configuration.

    Returns:
        Dataset summary and corrupted image records.

    Raises:
        DatasetValidationError: If the expected dataset structure is missing.
    """
    image_paths = collect_image_paths(config.paths.data_root)
    corrupted_images: list[dict[str, str]] = []

    for split in EXPECTED_SPLITS:
        for label in EXPECTED_CLASSES:
            for filepath in image_paths[split][label]:
                result = validate_image_file(filepath, split, label)
                if not result.is_valid:
                    corrupted_images.append(
                        {
                            "filepath": str(result.filepath),
                            "split": result.split,
                            "label": result.label,
                            "error": result.error or "unknown image decoding error",
                        }
                    )

    summary = build_dataset_summary(image_paths, corrupted_images)
    summary["data_root"] = str(config.paths.data_root)
    return summary, corrupted_images


def write_json(filepath: Path, payload: Any) -> None:
    """Write a JSON payload using UTF-8 encoding.

    Args:
        filepath: Output file path.
        payload: JSON-serializable payload.

    Raises:
        OSError: If the output file cannot be written.
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    filepath.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed command-line namespace.
    """
    parser = argparse.ArgumentParser(
        description="Check for corrupted images and inspect the class distribution."
    )
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to the YAML configuration file.",
    )
    return parser.parse_args()


def main() -> int:
    """Run dataset validation from the command line.

    Returns:
        Process exit code.
    """
    args = parse_args()
    try:
        config = load_and_prepare_config(args.config)
        summary, corrupted_images = validate_dataset(config)
        write_json(config.paths.metrics_dir / "dataset_summary.json", summary)
        write_json(config.paths.metrics_dir / "corrupted_images.json", corrupted_images)
    except (ConfigFileError, DatasetValidationError, OSError) as error:
        print(f"ERROR: {error}")
        return 1

    print("Dataset validation completed.")
    print(f"Summary file: {config.paths.metrics_dir / 'dataset_summary.json'}")
    print(f"Corrupted image report: {config.paths.metrics_dir / 'corrupted_images.json'}")
    for split in EXPECTED_SPLITS:
        split_summary = summary["splits"][split]
        print(
            f"- {split}: NORMAL={split_summary['NORMAL']}, "
            f"PNEUMONIA={split_summary['PNEUMONIA']}, total={split_summary['total']}"
        )
    print(f"Corrupted image count: {summary['corrupted_image_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
