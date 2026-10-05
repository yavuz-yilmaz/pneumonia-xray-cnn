"""Deterministic image preparation shared by training caches and inference."""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass, replace

import cv2
import numpy as np
from PIL import Image
from torch import Tensor
from torchvision.transforms import functional as functional_transforms

from src.data.dataset import ChestXRayDataset
from src.data.transforms import IMAGENET_NORMALIZATION_MEAN, IMAGENET_NORMALIZATION_STD


@dataclass(frozen=True)
class CheckpointImageTransform:
    """Pickle-safe deterministic transform for the complete checkpoint input contract."""

    image_size: int
    normalization_mean: tuple = IMAGENET_NORMALIZATION_MEAN
    normalization_std: tuple = IMAGENET_NORMALIZATION_STD
    preprocessing: str = "resize"

    def __call__(self, image: Image.Image) -> Tensor:
        """Apply preparation and normalization exactly as single-image inference."""
        prepared = prepare_image(image, self.image_size, self.preprocessing)
        tensor = functional_transforms.to_tensor(prepared)
        return functional_transforms.normalize(
            tensor, self.normalization_mean, self.normalization_std
        )

    @classmethod
    def from_metadata(cls, metadata: dict) -> CheckpointImageTransform:
        """Build an input transform from exported model metadata."""
        normalization = metadata.get("normalization", {})
        return cls(
            image_size=int(metadata["image_size"]),
            normalization_mean=tuple(normalization.get("mean", IMAGENET_NORMALIZATION_MEAN)),
            normalization_std=tuple(normalization.get("std", IMAGENET_NORMALIZATION_STD)),
            preprocessing=str(metadata.get("preprocessing", "resize")),
        )


def prepare_image(image: Image.Image, image_size: int, strategy: str = "resize") -> Image.Image:
    """Resize, optionally applying fixed local contrast normalization to grayscale.

    CLAHE parameters are fixed before fitting, not estimated using a test cohort.
    Applying it after resize gives identical processing for cache and inference.
    """
    if strategy not in {"resize", "clahe"}:
        raise ValueError(f"Unknown preprocessing strategy: {strategy}")
    image = functional_transforms.resize(image.convert("RGB"), [image_size, image_size])
    if strategy == "clahe":
        gray = np.asarray(image.convert("L"))
        normalized = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        image = Image.fromarray(normalized).convert("RGB")
    return image


def cache_prepared_images(dataset: ChestXRayDataset, image_size: int, strategy: str) -> None:
    """Use lossless prepared images to avoid repeating expensive source JPEG decoding.

    Source manifests remain unchanged. Cache keys include input SHA-256, image size,
    and preparation version. Training transforms still run afresh for every sample.
    """
    directory = dataset.manifest_path.parent / "image_cache" / f"{strategy}_{image_size}_v1"
    directory.mkdir(parents=True, exist_ok=True)
    with dataset.manifest_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != len(dataset.samples):
        raise ValueError("Dataset and cache manifest differ")
    prepared = []
    for row, sample in zip(rows, dataset.samples, strict=True):
        source_hash = row.get("sha256") or hashlib.sha256(sample.filepath.read_bytes()).hexdigest()
        target = directory / f"{source_hash}.png"
        if not target.exists():
            with Image.open(sample.filepath) as image:
                image = prepare_image(image, image_size, strategy)
            temporary = target.with_suffix(".tmp")
            image.save(temporary, format="PNG")
            temporary.replace(target)
        prepared.append(replace(sample, filepath=target))
    dataset.samples = prepared
