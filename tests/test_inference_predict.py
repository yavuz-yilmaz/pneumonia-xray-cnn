from __future__ import annotations

import io
from pathlib import Path

import pytest
import torch
from PIL import Image
from src.data.dataset import LABEL_TO_ID
from src.data.transforms import IMAGENET_NORMALIZATION_MEAN, IMAGENET_NORMALIZATION_STD
from src.inference.predict import InferenceError, load_model, predict_image, preprocess_image
from src.training.models import SimpleCNN


def test_preprocess_image_from_path_returns_batched_tensor(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.jpeg"
    Image.new("RGB", (48, 32), color=(120, 120, 120)).save(image_path)

    image_tensor = preprocess_image(image_path, image_size=64)

    assert image_tensor.shape == (1, 3, 64, 64)
    assert image_tensor.dtype == torch.float32


def test_preprocess_image_from_bytes_returns_batched_tensor() -> None:
    image_buffer = io.BytesIO()
    Image.new("RGB", (32, 32), color=(80, 80, 80)).save(image_buffer, format="PNG")

    image_tensor = preprocess_image(image_buffer.getvalue(), image_size=32)

    assert image_tensor.shape == (1, 3, 32, 32)


def test_preprocess_image_rejects_unsupported_extension(tmp_path: Path) -> None:
    text_path = tmp_path / "sample.txt"
    text_path.write_text("not an image", encoding="utf-8")

    with pytest.raises(InferenceError, match="Desteklenmeyen görüntü formatı"):
        preprocess_image(text_path)


def test_load_model_and_predict_image_return_expected_schema(tmp_path: Path) -> None:
    checkpoint_path = _create_test_checkpoint(tmp_path)
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (40, 40), color=(100, 100, 100)).save(image_path)

    loaded_model = load_model(checkpoint_path)
    image_tensor = preprocess_image(
        image_path,
        image_size=loaded_model.image_size,
        normalization_mean=loaded_model.normalization_mean,
        normalization_std=loaded_model.normalization_std,
    )
    prediction = predict_image(loaded_model, image_tensor)

    assert prediction["predicted_label"] in {"NORMAL", "PNEUMONIA"}
    assert 0.0 <= float(prediction["normal_probability"]) <= 1.0
    assert 0.0 <= float(prediction["pneumonia_probability"]) <= 1.0
    assert 0.0 <= float(prediction["confidence"]) <= 1.0
    assert prediction["model_version"] == checkpoint_path.name


def test_predict_image_rejects_unbatched_tensor(tmp_path: Path) -> None:
    checkpoint_path = _create_test_checkpoint(tmp_path)
    loaded_model = load_model(checkpoint_path)

    with pytest.raises(InferenceError, match="formatında olmalı"):
        predict_image(loaded_model, torch.zeros(3, 32, 32))


def test_load_model_rejects_missing_checkpoint(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Model checkpoint dosyası bulunamadı"):
        load_model(tmp_path / "missing.pt")


def _create_test_checkpoint(tmp_path: Path) -> Path:
    model = SimpleCNN(num_classes=2)
    checkpoint_path = tmp_path / "test_model.pt"
    torch.save(
        {
            "model_name": "simple_cnn",
            "model_state_dict": model.state_dict(),
            "label_mapping": LABEL_TO_ID,
            "id_to_label": {0: "NORMAL", 1: "PNEUMONIA"},
            "image_size": 32,
            "normalization": {
                "mean": IMAGENET_NORMALIZATION_MEAN,
                "std": IMAGENET_NORMALIZATION_STD,
            },
            "validation_metrics": {},
        },
        checkpoint_path,
    )
    return checkpoint_path
