"""Create manifest CSV files for model training without modifying raw images."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from src.core.config import ConfigFileError, ProjectConfig, load_and_prepare_config
from src.data.validate_dataset import (
    DatasetValidationError,
    collect_image_paths,
)


EXPECTED_SPLITS = ("train", "val", "test")
LABEL_TO_ID = {"NORMAL": 0, "PNEUMONIA": 1}
STRATIFIED_SPLIT_SUMMARY_FILENAME = "stratified_validation_split_summary.json"


class DatasetStandardizationError(RuntimeError):
    """Raised when manifest files cannot be produced safely."""


def build_manifest_rows(
    image_paths: dict[str, dict[str, list[Path]]],
) -> dict[str, list[dict[str, str | int]]]:
    """Build manifest rows for each dataset split.

    Args:
        image_paths: Nested mapping of split and label to image paths.

    Returns:
        Split names mapped to CSV row dictionaries.
    """
    manifests: dict[str, list[dict[str, str | int]]] = {}
    for split in EXPECTED_SPLITS:
        rows: list[dict[str, str | int]] = []
        for label, label_id in LABEL_TO_ID.items():
            for filepath in image_paths[split][label]:
                rows.append(
                    {
                        "filepath": str(filepath),
                        "split": split,
                        "label": label,
                        "label_id": label_id,
                    }
                )
        manifests[split] = sorted(rows, key=lambda row: (str(row["label"]), str(row["filepath"])))
    return manifests


def build_stratified_manifest_rows(
    image_paths: dict[str, dict[str, list[Path]]],
    validation_split_fraction: float,
    seed: int,
) -> dict[str, list[dict[str, str | int]]]:
    """Build manifests using a class-preserving validation split from raw train data.

    Args:
        image_paths: Nested mapping of split and label to image paths.
        validation_split_fraction: Fraction of each training class reserved for validation.
        seed: Deterministic shuffle seed.

    Returns:
        Split names mapped to CSV row dictionaries. The `train` and `val` outputs are
        derived from raw `train`; `test` remains the original raw test split.

    Raises:
        DatasetStandardizationError: If the requested split would empty a class.
    """
    train_rows: list[dict[str, str | int]] = []
    validation_rows: list[dict[str, str | int]] = []
    test_rows: list[dict[str, str | int]] = []

    for label, label_id in LABEL_TO_ID.items():
        class_paths = sorted(image_paths["train"][label])
        train_class_paths, validation_class_paths = split_class_paths(
            class_paths=class_paths,
            validation_split_fraction=validation_split_fraction,
            seed=seed + label_id,
            label=label,
        )
        train_rows.extend(
            build_rows_for_paths(
                paths=train_class_paths,
                split="train",
                label=label,
                label_id=label_id,
            )
        )
        validation_rows.extend(
            build_rows_for_paths(
                paths=validation_class_paths,
                split="val",
                label=label,
                label_id=label_id,
            )
        )
        test_rows.extend(
            build_rows_for_paths(
                paths=sorted(image_paths["test"][label]),
                split="test",
                label=label,
                label_id=label_id,
            )
        )

    return {
        "train": sorted(train_rows, key=lambda row: (str(row["label"]), str(row["filepath"]))),
        "val": sorted(validation_rows, key=lambda row: (str(row["label"]), str(row["filepath"]))),
        "test": sorted(test_rows, key=lambda row: (str(row["label"]), str(row["filepath"]))),
    }


def split_class_paths(
    class_paths: list[Path],
    validation_split_fraction: float,
    seed: int,
    label: str,
) -> tuple[list[Path], list[Path]]:
    """Split one class into deterministic train and validation path lists.

    Args:
        class_paths: Sorted image paths for a single class.
        validation_split_fraction: Fraction reserved for validation.
        seed: Deterministic shuffle seed.
        label: Human-readable class label used in error messages.

    Returns:
        Training paths and validation paths.

    Raises:
        DatasetStandardizationError: If the class is too small for a non-empty split.
    """
    import random

    if len(class_paths) < 2:
        message = (
            "A stratified validation split requires at least 2 training images per class. "
            f"Found {len(class_paths)} images for label='{label}'."
        )
        raise DatasetStandardizationError(message)

    validation_count = round(len(class_paths) * validation_split_fraction)
    validation_count = max(1, validation_count)
    validation_count = min(validation_count, len(class_paths) - 1)

    shuffled_paths = list(class_paths)
    random.Random(seed).shuffle(shuffled_paths)

    validation_paths = sorted(shuffled_paths[:validation_count])
    train_paths = sorted(shuffled_paths[validation_count:])
    return train_paths, validation_paths


def build_rows_for_paths(
    paths: list[Path],
    split: str,
    label: str,
    label_id: int,
) -> list[dict[str, str | int]]:
    """Build manifest row dictionaries for a list of image paths.

    Args:
        paths: Image file paths.
        split: Manifest split name to write.
        label: Human-readable class label.
        label_id: Numeric class identifier.

    Returns:
        Manifest rows with the standard columns.
    """
    return [
        {
            "filepath": str(filepath),
            "split": split,
            "label": label,
            "label_id": label_id,
        }
        for filepath in paths
    ]


def write_manifest_csv(output_path: Path, rows: list[dict[str, str | int]]) -> None:
    """Write one manifest CSV file.

    Args:
        output_path: CSV output path.
        rows: Manifest rows to write.

    Raises:
        DatasetStandardizationError: If a split has no rows.
        OSError: If the file cannot be written.
    """
    if not rows:
        message = f"Could not generate a manifest; no image rows: {output_path.name}"
        raise DatasetStandardizationError(message)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        fieldnames = ("filepath", "split", "label", "label_id")
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_split_summary(
    output_path: Path,
    manifests: dict[str, list[dict[str, str | int]]],
    raw_validation_rows: list[dict[str, str | int]],
    config: ProjectConfig,
) -> None:
    """Write a JSON summary explaining how the train/validation manifests were produced.

    Args:
        output_path: JSON output path.
        manifests: Generated train, validation, and test manifests.
        raw_validation_rows: Rows representing the untouched original raw validation split.
        config: Loaded project configuration.

    Raises:
        OSError: If the summary cannot be written.
    """
    summary = {
        "strategy": "stratified_split_from_raw_train"
        if config.data.use_stratified_validation_split
        else "raw_dataset_splits",
        "seed": config.training.seed,
        "validation_split_fraction": config.data.validation_split_fraction,
        "raw_validation_split_preserved_as": "raw_val_manifest.csv"
        if config.data.use_stratified_validation_split
        else None,
        "counts": {split: count_rows_by_label(rows) for split, rows in manifests.items()},
        "raw_validation_counts": count_rows_by_label(raw_validation_rows),
        "note": (
            "train_manifest.csv and val_manifest.csv were generated from raw train with class ratios preserved. "
            "The original raw val split is not used for training-time model selection because it is too small."
        )
        if config.data.use_stratified_validation_split
        else "Raw train/val/test folders were mirrored directly into manifests.",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def count_rows_by_label(rows: list[dict[str, str | int]]) -> dict[str, int]:
    """Count manifest rows by label.

    Args:
        rows: Manifest row dictionaries.

    Returns:
        Label names mapped to sample counts, plus a `total` entry.
    """
    counts = {label: 0 for label in LABEL_TO_ID}
    for row in rows:
        counts[str(row["label"])] += 1
    counts["total"] = sum(counts.values())
    return counts


def standardize_dataset(config: ProjectConfig) -> dict[str, Path]:
    """Create train, validation, and test manifest CSV files.

    Args:
        config: Loaded project configuration.

    Returns:
        Split names mapped to generated manifest paths.

    Raises:
        DatasetValidationError: If the expected dataset structure is missing.
        DatasetStandardizationError: If manifests cannot be generated.
        OSError: If output files cannot be written.
    """
    if config.data.use_group_disjoint_split:
        from src.data.clean_split import prepare_clean_split

        if config.data.validation_split_fraction != 0.2:
            raise DatasetStandardizationError(
                "Group-disjoint splitting uses a fixed 5-fold protocol; validation fraction must be 0.2"
            )
        prepare_clean_split(
            config.paths.data_root, config.paths.processed_data_dir, config.data.split_seed
        )
        return {
            **{
                split: config.paths.processed_data_dir / f"{split}_manifest.csv"
                for split in EXPECTED_SPLITS
            },
            "excluded": config.paths.processed_data_dir / "excluded_manifest.csv",
            "split_summary": config.paths.processed_data_dir / "split_audit.json",
        }

    image_paths = collect_image_paths(config.paths.data_root)
    raw_manifests = build_manifest_rows(image_paths)
    manifests = (
        build_stratified_manifest_rows(
            image_paths=image_paths,
            validation_split_fraction=config.data.validation_split_fraction,
            seed=config.training.seed,
        )
        if config.data.use_stratified_validation_split
        else raw_manifests
    )
    output_paths: dict[str, Path] = {}

    for split, rows in manifests.items():
        output_path = config.paths.processed_data_dir / f"{split}_manifest.csv"
        write_manifest_csv(output_path, rows)
        output_paths[split] = output_path

    if config.data.use_stratified_validation_split:
        raw_val_output_path = config.paths.processed_data_dir / "raw_val_manifest.csv"
        write_manifest_csv(raw_val_output_path, raw_manifests["val"])
        output_paths["raw_val"] = raw_val_output_path

    split_summary_path = config.paths.metrics_dir / STRATIFIED_SPLIT_SUMMARY_FILENAME
    write_split_summary(
        output_path=split_summary_path,
        manifests=manifests,
        raw_validation_rows=raw_manifests["val"],
        config=config,
    )
    output_paths["split_summary"] = split_summary_path

    return output_paths


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed command-line namespace.
    """
    parser = argparse.ArgumentParser(description="Generate manifest files for model training.")
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to the YAML configuration file.",
    )
    return parser.parse_args()


def main() -> int:
    """Run dataset standardization from the command line.

    Returns:
        Process exit code.
    """
    args = parse_args()
    try:
        config = load_and_prepare_config(args.config)
        output_paths = standardize_dataset(config)
    except (
        ConfigFileError,
        DatasetValidationError,
        DatasetStandardizationError,
        OSError,
    ) as error:
        print(f"ERROR: {error}")
        return 1

    print("Manifest files created.")
    for split, output_path in output_paths.items():
        print(f"- {split}: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
