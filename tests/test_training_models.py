import torch
from src.training.models import (
    ResNet18Ensemble,
    SimpleCNN,
    build_model,
    count_trainable_parameters,
    unfreeze_fine_tuning_layers,
)


def test_ensemble_softmax_equals_member_probability_average() -> None:
    class FixedLogits(torch.nn.Module):
        def __init__(self, logits: list[float]) -> None:
            super().__init__()
            self.register_buffer("logits", torch.tensor(logits))

        def forward(self, images: torch.Tensor) -> torch.Tensor:
            return self.logits.expand(len(images), -1)

    model = ResNet18Ensemble()
    model.members = torch.nn.ModuleList(
        [FixedLogits([0.0, 4.0]), FixedLogits([2.0, 0.0]), FixedLogits([1.0, 1.0])]
    )
    images = torch.zeros(2, 3, 32, 32)
    expected = torch.stack([member(images).softmax(1) for member in model.members]).mean(0)
    torch.testing.assert_close(model(images).softmax(1), expected)


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


def test_efficientnet_v2_head_trains_with_frozen_backbone() -> None:
    model = build_model("efficientnet_v2_s", 2, pretrained=False, freeze_backbone=True)
    model.eval()
    logits = model(torch.zeros(2, 3, 64, 64))
    torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1])).backward()
    assert logits.shape == (2, 2)
    assert model.classifier[1].weight.grad is not None
    assert all(parameter.grad is None for parameter in model.features.parameters())


def test_exported_xray_densenet_runs_without_loading_external_weights() -> None:
    model = build_model(
        model_name="xrv_densenet121", num_classes=2, pretrained=False, freeze_backbone=True
    )
    model.eval()
    with torch.no_grad():
        logits = model(torch.zeros(2, 3, 64, 64))
    assert logits.shape == (2, 2)
    assert torch.isfinite(logits).all()
    assert model.features.conv0.in_channels == 1
    assert count_trainable_parameters(model) == sum(
        p.numel() for p in model.classifier.parameters()
    )


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
