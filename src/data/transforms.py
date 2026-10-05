"""Image transform factories for chest X-ray training and evaluation."""

from __future__ import annotations

from collections.abc import Callable

from PIL import Image
from torch import Tensor
from torchvision import transforms


IMAGENET_NORMALIZATION_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_NORMALIZATION_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)


def build_train_transforms(image_size: int) -> Callable[[Image.Image], Tensor]:
    """Build controlled augmentation transforms for training images.

    Args:
        image_size: Square resize target in pixels.

    Returns:
        A callable that converts a PIL image to a normalized tensor.

    Raises:
        ValueError: If image size is not positive.
    """
    validate_image_size(image_size)
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=7),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_NORMALIZATION_MEAN,
                std=IMAGENET_NORMALIZATION_STD,
            ),
        ]
    )


def build_eval_transforms(image_size: int) -> Callable[[Image.Image], Tensor]:
    """Build deterministic transforms for validation, test, and inference images.

    Args:
        image_size: Square resize target in pixels.

    Returns:
        A callable that converts a PIL image to a normalized tensor.

    Raises:
        ValueError: If image size is not positive.
    """
    validate_image_size(image_size)
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_NORMALIZATION_MEAN,
                std=IMAGENET_NORMALIZATION_STD,
            ),
        ]
    )


def validate_image_size(image_size: int) -> None:
    """Validate image resize dimension.

    Args:
        image_size: Candidate square resize target in pixels.

    Raises:
        ValueError: If image size is not positive.
    """
    if image_size <= 0:
        message = f"image_size must be a positive integer; received: {image_size}"
        raise ValueError(message)
