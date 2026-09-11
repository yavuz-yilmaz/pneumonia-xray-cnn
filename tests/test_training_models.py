import torch
from src.training.models import (
    SimpleCNN,
    build_model,
    count_trainable_parameters,
    unfreeze_fine_tuning_layers,
)


def test_simple_cnn_forward_shape() -> None:
    model = SimpleCNN(num_classes=2)
    model.eval()
    images = torch.randn(2, 3, 64, 64)

    with torch.no_grad():
        logits = model(images)

    assert logits.shape == (2, 2)


def test_build_model_returns_simple_cnn() -> None:
    model = build_model(model_name="simple_cnn", num_classes=2)

    assert isinstance(model, SimpleCNN)
    assert count_trainable_parameters(model) > 0


def test_build_model_returns_frozen_resnet18_classifier() -> None:
    model = build_model(
        model_name="resnet18",
        num_classes=2,
        pretrained=False,
        freeze_backbone=True,
    )
    model.eval()
    images = torch.randn(2, 3, 64, 64)

    with torch.no_grad():
        logits = model(images)

    assert logits.shape == (2, 2)
    assert count_trainable_parameters(model) == sum(
        parameter.numel() for parameter in model.fc.parameters()
    )


def test_unfreeze_fine_tuning_layers_opens_resnet18_layer4() -> None:
    model = build_model(
        model_name="resnet18",
        num_classes=2,
        pretrained=False,
        freeze_backbone=True,
    )

    frozen_parameter_count = count_trainable_parameters(model)
    unfreeze_fine_tuning_layers(model, "resnet18")

    assert count_trainable_parameters(model) > frozen_parameter_count
    assert any(parameter.requires_grad for parameter in model.layer4.parameters())
