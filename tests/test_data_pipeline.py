import csv
from pathlib import Path

import pytest
import torch
from PIL import Image
from src.data.dataloader import build_weighted_sampler, calculate_class_weights
from src.data.dataset import ChestXRayDataset, XRayDatasetError
from src.data.standardize_dataset import (
    build_stratified_manifest_rows,
    count_rows_by_label,
)
from src.data.transforms import build_eval_transforms, build_train_transforms
from torch.utils.data import DataLoader


def create_tiny_manifest(tmp_path: Path) -> Path:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    rows: list[dict[str, str | int]] = []
    sample_specs = (
        ("normal_1.jpeg", "NORMAL", 0, (32, 32, 32)),
        ("pneumonia_1.jpeg", "PNEUMONIA", 1, (180, 180, 180)),
        ("pneumonia_2.jpeg", "PNEUMONIA", 1, (220, 220, 220)),
    )

    for filename, label, label_id, color in sample_specs:
        image_path = image_dir / filename
        Image.new("RGB", (48, 40), color=color).save(image_path, format="JPEG")
        rows.append(
            {
                "filepath": str(image_path),
                "split": "train",
                "label": label,
                "label_id": label_id,
            }
        )

    manifest_path = tmp_path / "train_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as manifest_file:
        writer = csv.DictWriter(
            manifest_file,
            fieldnames=("filepath", "split", "label", "label_id"),
        )
        writer.writeheader()
        writer.writerows(rows)

    return manifest_path


def test_chest_xray_dataset_length_and_tensor_shape(tmp_path: Path) -> None:
    manifest_path = create_tiny_manifest(tmp_path)
    dataset = ChestXRayDataset(manifest_path, transform=build_eval_transforms(image_size=64))

    image_tensor, label_id = dataset[0]

    assert len(dataset) == 3
    assert image_tensor.shape == (3, 64, 64)
    assert image_tensor.dtype == torch.float32
    assert label_id in {0, 1}


def test_dataloader_can_return_one_batch(tmp_path: Path) -> None:
    manifest_path = create_tiny_manifest(tmp_path)
    dataset = ChestXRayDataset(manifest_path, transform=build_train_transforms(image_size=32))
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)

    image_batch, label_batch = next(iter(dataloader))

    assert image_batch.shape == (2, 3, 32, 32)
    assert label_batch.shape == (2,)
    assert set(label_batch.tolist()).issubset({0, 1})


def test_eval_transforms_are_deterministic() -> None:
    image = Image.new("RGB", (48, 40), color=(128, 128, 128))
    transform = build_eval_transforms(image_size=64)

    first_tensor = transform(image)
    second_tensor = transform(image)

    assert first_tensor.shape == (3, 64, 64)
    assert torch.equal(first_tensor, second_tensor)


def test_train_transforms_return_finite_tensor_with_expected_shape() -> None:
    image = Image.new("RGB", (48, 40), color=(128, 128, 128))
    transform = build_train_transforms(image_size=32)

    image_tensor = transform(image)

    assert image_tensor.shape == (3, 32, 32)
    assert image_tensor.dtype == torch.float32
    assert torch.isfinite(image_tensor).all()


def test_class_weights_reflect_manifest_imbalance(tmp_path: Path) -> None:
    manifest_path = create_tiny_manifest(tmp_path)
    dataset = ChestXRayDataset(manifest_path, transform=build_eval_transforms(image_size=32))

    class_weights = calculate_class_weights(dataset)
    sampler = build_weighted_sampler(dataset)

    assert class_weights.shape == (2,)
    assert class_weights[0] > class_weights[1]
    assert sampler.num_samples == len(dataset)


def test_dataset_rejects_inconsistent_label_mapping(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.jpeg"
    Image.new("RGB", (16, 16), color=(255, 255, 255)).save(image_path, format="JPEG")
    manifest_path = tmp_path / "bad_manifest.csv"
    manifest_path.write_text(
        f"filepath,split,label,label_id\n{image_path},train,NORMAL,1\n",
        encoding="utf-8",
    )

    with pytest.raises(XRayDatasetError, match="Label mapping tutarsız"):
        ChestXRayDataset(manifest_path, transform=build_eval_transforms(image_size=32))


def test_stratified_manifest_split_preserves_class_counts() -> None:
    image_paths = {
        "train": {
            "NORMAL": [Path(f"normal_{index}.jpeg") for index in range(10)],
            "PNEUMONIA": [Path(f"pneumonia_{index}.jpeg") for index in range(30)],
        },
        "val": {
            "NORMAL": [Path("raw_val_normal.jpeg")],
            "PNEUMONIA": [Path("raw_val_pneumonia.jpeg")],
        },
        "test": {
            "NORMAL": [Path(f"test_normal_{index}.jpeg") for index in range(2)],
            "PNEUMONIA": [Path(f"test_pneumonia_{index}.jpeg") for index in range(4)],
        },
    }

    manifests = build_stratified_manifest_rows(
        image_paths=image_paths,
        validation_split_fraction=0.2,
        seed=42,
    )

    assert count_rows_by_label(manifests["train"]) == {
        "NORMAL": 8,
        "PNEUMONIA": 24,
        "total": 32,
    }
    assert count_rows_by_label(manifests["val"]) == {
        "NORMAL": 2,
        "PNEUMONIA": 6,
        "total": 8,
    }
    assert count_rows_by_label(manifests["test"]) == {
        "NORMAL": 2,
        "PNEUMONIA": 4,
        "total": 6,
    }
    assert {row["split"] for row in manifests["val"]} == {"val"}
