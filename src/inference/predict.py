"""Single-image inference pipeline for trained pneumonia X-ray classifiers."""

from __future__ import annotations

import io
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError
from torch import Tensor, nn
from torchvision import transforms

from src.data.dataset import ID_TO_LABEL, LABEL_TO_ID, SUPPORTED_IMAGE_EXTENSIONS
from src.data.image_preparation import prepare_image
from src.data.transforms import IMAGENET_NORMALIZATION_MEAN, IMAGENET_NORMALIZATION_STD
from src.training.models import build_model

PNEUMONIA_LABEL_ID = LABEL_TO_ID["PNEUMONIA"]
NORMAL_LABEL_ID = LABEL_TO_ID["NORMAL"]
DEFAULT_IMAGE_SIZE = 224
DEFAULT_PNEUMONIA_THRESHOLD = 0.70


class InferenceError(RuntimeError):
    """Raised when a single-image inference operation cannot be completed."""


@dataclass(frozen=True)
class LoadedModel:
    """Trained model and checkpoint metadata required for inference.

    Attributes:
        model: PyTorch model loaded with trained weights.
        checkpoint_path: Filesystem path of the loaded checkpoint.
        model_name: Architecture name stored in the checkpoint.
        image_size: Square image size expected by the model.
        normalization_mean: Per-channel normalization mean.
        normalization_std: Per-channel normalization standard deviation.
        label_mapping: Mapping from class label to numeric class identifier.
        device: Device where the model is loaded.
    """

    model: nn.Module
    checkpoint_path: Path
    model_name: str
    image_size: int
    normalization_mean: tuple[float, float, float]
    normalization_std: tuple[float, float, float]
    label_mapping: dict[str, int]
    device: torch.device
    decision_threshold: float = DEFAULT_PNEUMONIA_THRESHOLD
    model_version: str = ""
    preprocessing: str = "resize"


@dataclass(frozen=True)
class PredictionResult:
    """JSON-serializable prediction result for one chest X-ray image.

    Attributes:
        predicted_label: Predicted class label.
        pneumonia_probability: Probability assigned to the PNEUMONIA class.
        normal_probability: Probability assigned to the NORMAL class.
        confidence: Probability of the predicted class.
        model_version: Checkpoint filename used for inference.
    """

    predicted_label: str
    pneumonia_probability: float
    normal_probability: float
    confidence: float
    model_version: str

    def to_dict(self) -> dict[str, str | float]:
        """Convert the prediction to a plain JSON-compatible dictionary.

        Returns:
            Prediction fields as primitive JSON-compatible values.
        """
        return {
            "predicted_label": self.predicted_label,
            "pneumonia_probability": self.pneumonia_probability,
            "normal_probability": self.normal_probability,
            "confidence": self.confidence,
            "model_version": self.model_version,
        }


def load_model(checkpoint_path: str | Path) -> LoadedModel:
    """Load a trained checkpoint for single-image inference.

    Args:
        checkpoint_path: Path to a checkpoint created by the training workflow.

    Returns:
        Loaded model bundle containing the model and inference metadata.

    Raises:
        FileNotFoundError: If the checkpoint path does not exist.
        InferenceError: If the checkpoint is malformed or incompatible.
    """
    resolved_checkpoint_path = Path(checkpoint_path)
    if not resolved_checkpoint_path.is_file():
        message = (
            f"Model checkpoint file not found: {resolved_checkpoint_path}. "
            "Train a model first or provide the correct `--model` path."
        )
        raise FileNotFoundError(message)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        checkpoint = torch.load(resolved_checkpoint_path, map_location=device, weights_only=False)
    except (OSError, RuntimeError, ValueError) as error:
        message = (
            f"Could not load the model checkpoint file: {resolved_checkpoint_path}. Error: {error}"
        )
        raise InferenceError(message) from error

    if not isinstance(checkpoint, dict):
        message = f"The checkpoint is not a dictionary: {resolved_checkpoint_path}"
        raise InferenceError(message)

    required_keys = {"model_name", "model_state_dict", "label_mapping", "image_size"}
    missing_keys = required_keys.difference(checkpoint)
    if missing_keys:
        formatted_keys = ", ".join(sorted(missing_keys))
        message = f"The checkpoint is missing required inference fields: {formatted_keys}"
        raise InferenceError(message)

    label_mapping = checkpoint["label_mapping"]
    if label_mapping != LABEL_TO_ID:
        message = (
            "The checkpoint label mapping does not match the expected values. "
            f"Expected={LABEL_TO_ID}, received={label_mapping}"
        )
        raise InferenceError(message)

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
        message = f"Checkpoint weights are incompatible with `{model_name}`: {error}"
        raise InferenceError(message) from error

    image_size = _parse_image_size(checkpoint["image_size"])
    normalization_mean, normalization_std = _parse_normalization(checkpoint)
    try:
        decision_threshold = float(
            checkpoint.get("decision_threshold", DEFAULT_PNEUMONIA_THRESHOLD)
        )
    except (TypeError, ValueError) as error:
        raise InferenceError("Checkpoint decision_threshold must be numeric") from error
    if not 0.0 <= decision_threshold <= 1.0:
        raise InferenceError("Checkpoint decision_threshold must be finite and in [0, 1]")
    preprocessing = str(checkpoint.get("preprocessing", "resize"))
    if preprocessing not in {"resize", "clahe"}:
        raise InferenceError(f"Unknown checkpoint preprocessing: {preprocessing}")
    model.to(device)
    model.eval()

    return LoadedModel(
        model=model,
        checkpoint_path=resolved_checkpoint_path,
        model_name=model_name,
        image_size=image_size,
        normalization_mean=normalization_mean,
        normalization_std=normalization_std,
        label_mapping=label_mapping,
        device=device,
        decision_threshold=decision_threshold,
        model_version=str(checkpoint.get("model_version", resolved_checkpoint_path.name)),
        preprocessing=preprocessing,
    )


def preprocess_image(
    image_path_or_bytes: str | Path | bytes,
    image_size: int = DEFAULT_IMAGE_SIZE,
    normalization_mean: tuple[float, float, float] = IMAGENET_NORMALIZATION_MEAN,
    normalization_std: tuple[float, float, float] = IMAGENET_NORMALIZATION_STD,
    preprocessing: str = "resize",
) -> Tensor:
    """Load, validate, and transform one image for model inference.

    Args:
        image_path_or_bytes: Image path or raw encoded image bytes.
        image_size: Square resize target in pixels.
        normalization_mean: Per-channel normalization mean.
        normalization_std: Per-channel normalization standard deviation.

    Returns:
        Normalized tensor with shape `(1, 3, image_size, image_size)`.

    Raises:
        FileNotFoundError: If an image path does not exist.
        InferenceError: If the image is unsupported, corrupt, or cannot be transformed.
    """
    if image_size <= 0:
        message = f"image_size must be a positive integer; received: {image_size}"
        raise InferenceError(message)

    try:
        pil_image = _open_image(image_path_or_bytes)
    except FileNotFoundError:
        raise
    except InferenceError:
        raise

    pil_image = prepare_image(pil_image, image_size, preprocessing)
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=normalization_mean, std=normalization_std),
        ]
    )
    try:
        image_tensor = transform(pil_image)
    except (RuntimeError, ValueError, TypeError) as error:
        message = f"Could not process the image during the inference transform. Error: {error}"
        raise InferenceError(message) from error

    return image_tensor.unsqueeze(0)


def predict_image(model: LoadedModel, image: Tensor) -> dict[str, str | float]:
    """Predict the class of a preprocessed single X-ray image.

    Args:
        model: Loaded model bundle returned by `load_model`.
        image: Preprocessed tensor with shape `(1, 3, height, width)`.

    Returns:
        Prediction dictionary containing labels, probabilities, confidence, and model version.

    Raises:
        InferenceError: If the image tensor has an invalid shape or inference fails.
    """
    if image.ndim != 4 or image.shape[0] != 1 or image.shape[1] != 3:
        message = (
            "The inference image tensor must have shape `(1, 3, height, width)`; "
            f"received shape: {tuple(image.shape)}"
        )
        raise InferenceError(message)

    try:
        with torch.no_grad():
            logits = model.model(image.to(model.device))
            probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu()
    except RuntimeError as error:
        message = f"Could not generate a model prediction. Error: {error}"
        raise InferenceError(message) from error

    if probabilities.numel() != len(ID_TO_LABEL):
        message = (
            f"The model must return {len(ID_TO_LABEL)} class probabilities; "
            f"received probability count: {probabilities.numel()}"
        )
        raise InferenceError(message)

    pneumonia_probability = float(probabilities[PNEUMONIA_LABEL_ID].item())
    normal_probability = float(probabilities[NORMAL_LABEL_ID].item())
    predicted_label_id = (
        PNEUMONIA_LABEL_ID if pneumonia_probability >= model.decision_threshold else NORMAL_LABEL_ID
    )
    predicted_label = ID_TO_LABEL[predicted_label_id]
    confidence = float(probabilities[predicted_label_id].item())
    result = PredictionResult(
        predicted_label=predicted_label,
        pneumonia_probability=pneumonia_probability,
        normal_probability=normal_probability,
        confidence=confidence,
        model_version=model.model_version or model.checkpoint_path.name,
    )
    return result.to_dict()


def predict_image_file(
    image_path: str | Path,
    checkpoint_path: str | Path,
) -> dict[str, str | float]:
    """Load a checkpoint and predict one image path in a single call.

    Args:
        image_path: Path to a `.jpeg`, `.jpg`, or `.png` image.
        checkpoint_path: Path to a trained model checkpoint.

    Returns:
        Prediction dictionary suitable for JSON serialization.

    Raises:
        FileNotFoundError: If the image or checkpoint path is missing.
        InferenceError: If loading, preprocessing, or prediction fails.
    """
    loaded_model = load_model(checkpoint_path)
    image_tensor = preprocess_image(
        image_path,
        image_size=loaded_model.image_size,
        normalization_mean=loaded_model.normalization_mean,
        normalization_std=loaded_model.normalization_std,
        preprocessing=loaded_model.preprocessing,
    )
    return predict_image(loaded_model, image_tensor)


def _open_image(image_path_or_bytes: str | Path | bytes) -> Image.Image:
    if isinstance(image_path_or_bytes, bytes):
        if not image_path_or_bytes:
            message = "Image bytes are empty; provide a valid JPEG, JPG, or PNG image."
            raise InferenceError(message)
        try:
            with Image.open(io.BytesIO(image_path_or_bytes)) as image:
                return image.convert("RGB")
        except (OSError, UnidentifiedImageError) as error:
            message = f"Image bytes are unreadable or corrupted. Error: {error}"
            raise InferenceError(message) from error

    image_path = Path(image_path_or_bytes)
    if not image_path.is_file():
        message = f"Image file not found: {image_path}"
        raise FileNotFoundError(message)
    if image_path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
        message = (
            f"Unsupported image format: {image_path.suffix}. "
            f"Supported extensions: {', '.join(sorted(SUPPORTED_IMAGE_EXTENSIONS))}"
        )
        raise InferenceError(message)

    try:
        with Image.open(image_path) as image:
            return image.convert("RGB")
    except (OSError, UnidentifiedImageError) as error:
        message = f"The image file is unreadable or corrupted: {image_path}. Error: {error}"
        raise InferenceError(message) from error


def _parse_image_size(raw_image_size: object) -> int:
    try:
        image_size = int(raw_image_size)
    except (TypeError, ValueError) as error:
        message = f"Checkpoint image_size must be an integer; received: {raw_image_size}"
        raise InferenceError(message) from error
    if image_size <= 0:
        message = f"Checkpoint image_size must be positive; received: {image_size}"
        raise InferenceError(message)
    return image_size


def _parse_normalization(
    checkpoint: dict[str, object],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    normalization = checkpoint.get("normalization", {})
    if not isinstance(normalization, dict):
        message = "Checkpoint normalization must be a dictionary."
        raise InferenceError(message)

    mean = _parse_float_triplet(
        normalization.get("mean", IMAGENET_NORMALIZATION_MEAN),
        field_name="normalization.mean",
    )
    std = _parse_float_triplet(
        normalization.get("std", IMAGENET_NORMALIZATION_STD),
        field_name="normalization.std",
    )
    if any(value <= 0.0 for value in std):
        message = f"Checkpoint normalization.std values must be positive; received: {std}"
        raise InferenceError(message)
    return mean, std


def _parse_float_triplet(raw_values: object, *, field_name: str) -> tuple[float, float, float]:
    if (
        not isinstance(raw_values, Sequence)
        or isinstance(raw_values, str | bytes)
        or len(raw_values) != 3
    ):
        message = f"Checkpoint {field_name} must be a three-element list or tuple."
        raise InferenceError(message)
    try:
        values = tuple(float(value) for value in raw_values)
    except (TypeError, ValueError) as error:
        message = f"Checkpoint {field_name} must contain only numeric values."
        raise InferenceError(message) from error
    return values[0], values[1], values[2]
