"""Training cache and deployed preprocessing must produce identical tensors."""

import csv
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from src.data.dataset import ChestXRayDataset
from src.data.image_preparation import CheckpointImageTransform, cache_prepared_images
from src.data.transforms import build_eval_transforms
from src.inference.predict import preprocess_image


@pytest.mark.parametrize("strategy", ["resize", "clahe"])
def test_prepared_cache_matches_inference(tmp_path: Path, strategy: str) -> None:
    image_path = tmp_path / "sample.png"
    pixels = np.random.default_rng(42).integers(0, 256, (49, 37, 3), dtype=np.uint8)
    Image.fromarray(pixels).save(image_path)
    manifest = tmp_path / "train_manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["filepath", "split", "label", "label_id"])
        writer.writeheader()
        writer.writerow(
            {"filepath": str(image_path), "split": "train", "label": "NORMAL", "label_id": 0}
        )
    dataset = ChestXRayDataset(manifest, build_eval_transforms(32))
    cache_prepared_images(dataset, 32, strategy)
    training_tensor, _label = dataset[0]
    serving_tensor = preprocess_image(image_path, image_size=32, preprocessing=strategy)
    assert torch.equal(training_tensor, serving_tensor.squeeze(0))


@pytest.mark.parametrize("strategy", ["resize", "clahe"])
def test_checkpoint_transform_matches_xray_model_inference(tmp_path: Path, strategy: str) -> None:
    path = tmp_path / "xray.png"
    pixels = np.random.default_rng(7).integers(0, 256, (43, 59, 3), dtype=np.uint8)
    image = Image.fromarray(pixels)
    image.save(path)
    metadata = {
        "image_size": 32,
        "normalization": {"mean": (0.5,) * 3, "std": (1 / 2048,) * 3},
        "preprocessing": strategy,
    }
    transform = CheckpointImageTransform.from_metadata(metadata)
    expected = preprocess_image(
        path, 32, (0.5,) * 3, (1 / 2048,) * 3, preprocessing=strategy
    ).squeeze(0)
    assert torch.equal(transform(image), expected)
