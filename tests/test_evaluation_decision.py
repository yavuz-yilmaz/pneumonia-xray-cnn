"""Serving and batch evaluation must interpret the operating point identically."""

from pathlib import Path

import pytest
import torch
from src.evaluation.evaluate import collect_test_predictions
from src.inference.predict import LoadedModel, predict_image
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class FixedProbabilityModel(nn.Module):
    """Emit known class scores to exercise threshold boundary behavior."""

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        probability = images[:, 0, 0, 0]
        return torch.stack((1 - probability, probability), dim=1).log()


def test_batch_and_serving_agree_on_non_default_threshold() -> None:
    images = torch.zeros(3, 3, 2, 2)
    images[:, 0, 0, 0] = torch.tensor([0.4, 0.6, 0.9])
    model = FixedProbabilityModel()
    loaded = LoadedModel(
        model=model,
        checkpoint_path=Path("example.pt"),
        model_name="test",
        image_size=2,
        normalization_mean=(0.0, 0.0, 0.0),
        normalization_std=(1.0, 1.0, 1.0),
        label_mapping={"NORMAL": 0, "PNEUMONIA": 1},
        device=torch.device("cpu"),
        decision_threshold=0.7,
    )
    dataset = TensorDataset(images, torch.tensor([0, 0, 1]))
    batch = collect_test_predictions(
        model=model,
        dataloader=DataLoader(dataset, batch_size=3),
        device=loaded.device,
        decision_threshold=loaded.decision_threshold,
    )
    single = [predict_image(loaded, sample.unsqueeze(0)) for sample in images]
    assert batch.predictions == [0, 0, 1]
    assert [entry["predicted_label"] for entry in single] == ["NORMAL", "NORMAL", "PNEUMONIA"]
    assert single[1]["confidence"] == pytest.approx(0.4)
    assert batch.probabilities == pytest.approx(
        [entry["pneumonia_probability"] for entry in single]
    )
