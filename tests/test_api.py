from __future__ import annotations

import io
from pathlib import Path

import torch
from app.api.main import MEDICAL_WARNING, create_app
from fastapi.testclient import TestClient
from PIL import Image
from src.data.dataset import LABEL_TO_ID
from src.data.transforms import IMAGENET_NORMALIZATION_MEAN, IMAGENET_NORMALIZATION_STD
from src.training.models import SimpleCNN


def test_health_endpoint_reports_loaded_model(tmp_path: Path) -> None:
    checkpoint_path = _create_test_checkpoint(tmp_path)
    test_app = create_app(checkpoint_path=checkpoint_path)

    with TestClient(test_app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_model_info_endpoint_returns_checkpoint_metadata(tmp_path: Path) -> None:
    checkpoint_path = _create_test_checkpoint(tmp_path)
    test_app = create_app(checkpoint_path=checkpoint_path)

    with TestClient(test_app) as client:
        response = client.get("/model-info")

    assert response.status_code == 200
    body = response.json()
    assert body["model_name"] == "simple_cnn"
    assert body["model_version"] == checkpoint_path.name
    assert body["checkpoint_path"] == str(checkpoint_path)
    assert body["image_size"] == 32
    assert body["label_mapping"] == LABEL_TO_ID
    assert body["warning"] == MEDICAL_WARNING


def test_predict_endpoint_returns_prediction_for_valid_image(tmp_path: Path) -> None:
    checkpoint_path = _create_test_checkpoint(tmp_path)
    image_bytes = _create_png_bytes()
    test_app = create_app(checkpoint_path=checkpoint_path)

    with TestClient(test_app) as client:
        response = client.post(
            "/predict",
            files={"file": ("sample.png", image_bytes, "image/png")},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["predicted_label"] in {"NORMAL", "PNEUMONIA"}
    assert 0.0 <= body["normal_probability"] <= 1.0
    assert 0.0 <= body["pneumonia_probability"] <= 1.0
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["warning"] == MEDICAL_WARNING


def test_predict_endpoint_rejects_invalid_file(tmp_path: Path) -> None:
    checkpoint_path = _create_test_checkpoint(tmp_path)
    test_app = create_app(checkpoint_path=checkpoint_path)

    with TestClient(test_app) as client:
        response = client.post(
            "/predict",
            files={"file": ("sample.txt", b"not an image", "text/plain")},
        )

    assert response.status_code == 415
    assert "Desteklenmeyen görüntü formatı" in response.json()["detail"]


def test_predict_endpoint_rejects_missing_file_field(tmp_path: Path) -> None:
    checkpoint_path = _create_test_checkpoint(tmp_path)
    test_app = create_app(checkpoint_path=checkpoint_path)

    with TestClient(test_app) as client:
        response = client.post("/predict")

    assert response.status_code == 400
    assert "multipart `file` alanı zorunludur" in response.json()["detail"]


def test_predict_endpoint_rejects_corrupted_image_bytes(tmp_path: Path) -> None:
    checkpoint_path = _create_test_checkpoint(tmp_path)
    test_app = create_app(checkpoint_path=checkpoint_path)

    with TestClient(test_app) as client:
        response = client.post(
            "/predict",
            files={"file": ("sample.png", b"not an image", "image/png")},
        )

    assert response.status_code == 400
    assert "Görüntü işlenemedi" in response.json()["detail"]


def _create_test_checkpoint(tmp_path: Path) -> Path:
    model = SimpleCNN(num_classes=2)
    checkpoint_path = tmp_path / "api_test_model.pt"
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


def _create_png_bytes() -> bytes:
    image_buffer = io.BytesIO()
    Image.new("RGB", (40, 40), color=(96, 96, 96)).save(image_buffer, format="PNG")
    return image_buffer.getvalue()
