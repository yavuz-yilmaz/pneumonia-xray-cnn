"""Model definitions for chest X-ray pneumonia classification."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch import Tensor, nn
from torchvision.models import (
    EfficientNet_B0_Weights,
    EfficientNet_V2_S_Weights,
    MobileNet_V3_Small_Weights,
    ResNet18_Weights,
    densenet121,
    efficientnet_b0,
    efficientnet_v2_s,
    mobilenet_v3_small,
    resnet18,
)


class ResNet18Ensemble(nn.Module):
    """Three independent ResNet18 members with fixed probability averaging."""

    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        self.members = nn.ModuleList(
            [
                build_resnet18(num_classes=num_classes, pretrained=False, freeze_backbone=False)
                for _ in range(3)
            ]
        )

    def forward(self, images: Tensor) -> Tensor:
        """Return log probabilities so existing softmax-based consumers agree."""
        probabilities = torch.stack([member(images).softmax(1) for member in self.members]).mean(0)
        return probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny).log()


class XRayDenseNet121(nn.Module):
    """Torchvision-compatible DenseNet with chest-X-ray-pretrained feature weights.

    Inputs are RGB-repeated grayscale in the XRV [-1024, 1024] range. Exported
    checkpoints need only torchvision; TorchXRayVision is a training-only loader.
    """

    def __init__(self, num_classes: int = 2, pretrained: bool = False) -> None:
        super().__init__()
        base = densenet121(weights=None)
        base.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.features = base.features
        self.classifier = nn.Linear(1024, num_classes)
        if pretrained:
            import torchxrayvision as xrv

            source = xrv.models.DenseNet(weights="densenet121-res224-all")
            self.features.load_state_dict(source.features.state_dict(), strict=True)
            self.pretraining_metadata = {
                "provider": "TorchXRayVision",
                "weights": "densenet121-res224-all",
                "sha256": hashlib.sha256(
                    Path(source.weights_filename_local).read_bytes()
                ).hexdigest(),
            }

    def extract_features(self, images: Tensor) -> Tensor:
        """Produce the external pretrained representation without its disease heads."""
        features = self.features(images.mean(dim=1, keepdim=True))
        return nn.functional.adaptive_avg_pool2d(nn.functional.relu(features), (1, 1)).flatten(1)

    def forward(self, images: Tensor) -> Tensor:
        """Return binary class logits."""
        return self.classifier(self.extract_features(images))


class SimpleCNN(nn.Module):
    """Baseline convolutional neural network for binary X-ray classification.

    Args:
        num_classes: Number of output classes.
        dropout_probability: Dropout probability used in convolutional and classifier blocks.

    Raises:
        ValueError: If `num_classes` is smaller than 2 or dropout is outside [0, 1).
    """

    def __init__(self, num_classes: int = 2, dropout_probability: float = 0.3) -> None:
        super().__init__()
        if num_classes < 2:
            message = f"num_classes must be at least 2; received: {num_classes}"
            raise ValueError(message)
        if not 0.0 <= dropout_probability < 1.0:
            message = (
                "dropout_probability must be in [0.0, 1.0); "
                f"received value: {dropout_probability}"
            )
            raise ValueError(message)

        self.features = nn.Sequential(
            self._build_conv_block(in_channels=3, out_channels=32, dropout_probability=0.0),
            self._build_conv_block(
                in_channels=32,
                out_channels=64,
                dropout_probability=dropout_probability,
            ),
            self._build_conv_block(
                in_channels=64,
                out_channels=128,
                dropout_probability=dropout_probability,
            ),
            self._build_conv_block(
                in_channels=128,
                out_channels=256,
                dropout_probability=dropout_probability,
            ),
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_probability),
            nn.Linear(128, num_classes),
        )

    def forward(self, images: Tensor) -> Tensor:
        """Run a forward pass.

        Args:
            images: Batch tensor with shape `(batch_size, 3, height, width)`.

        Returns:
            Raw class logits with shape `(batch_size, num_classes)`.
        """
        feature_maps = self.features(images)
        pooled_features = self.global_pool(feature_maps)
        return self.classifier(pooled_features)

    @staticmethod
    def _build_conv_block(
        in_channels: int,
        out_channels: int,
        dropout_probability: float,
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=dropout_probability),
        )


def build_model(
    model_name: str,
    num_classes: int,
    *,
    pretrained: bool = True,
    freeze_backbone: bool = False,
) -> nn.Module:
    """Build a supported classification model.

    Args:
        model_name: Model architecture name.
        num_classes: Number of output classes.
        pretrained: Whether to initialize supported transfer-learning models with ImageNet weights.
        freeze_backbone: Whether to freeze feature extractor parameters after model creation.

    Returns:
        Instantiated PyTorch module.

    Raises:
        ValueError: If the model name is unsupported.
    """
    normalized_model_name = model_name.strip().lower()
    if normalized_model_name == "resnet18_ensemble3":
        if pretrained:
            raise ValueError("Ensembles must be exported from three completed runs")
        return ResNet18Ensemble(num_classes)
    if normalized_model_name == "efficientnet_v2_s":
        weights = EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
        model = efficientnet_v2_s(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        if weights is not None:
            model.pretraining_metadata = {
                "provider": "torchvision",
                "weights": str(weights),
                "url": weights.url,
            }
        if freeze_backbone:
            freeze_feature_extractor(model)
        return model
    if normalized_model_name == "xrv_densenet121":
        model = XRayDenseNet121(num_classes=num_classes, pretrained=pretrained)
        if freeze_backbone:
            freeze_feature_extractor(model)
        return model
    if normalized_model_name == "simple_cnn":
        return SimpleCNN(num_classes=num_classes)
    if normalized_model_name == "resnet18":
        return build_resnet18(
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
        )
    if normalized_model_name == "efficientnet_b0":
        return build_efficientnet_b0(
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
        )
    if normalized_model_name == "mobilenet_v3_small":
        return build_mobilenet_v3_small(
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
        )

    message = (
        "Supported model names: simple_cnn, resnet18, efficientnet_b0, "
        f"mobilenet_v3_small. Received value: {model_name}"
    )
    raise ValueError(message)


def build_resnet18(
    *,
    num_classes: int,
    pretrained: bool,
    freeze_backbone: bool,
) -> nn.Module:
    """Build a ResNet18 transfer-learning classifier.

    Args:
        num_classes: Number of output classes.
        pretrained: Whether to use ImageNet weights.
        freeze_backbone: Whether to freeze all layers except the final classifier.

    Returns:
        ResNet18 model with a binary-compatible fully connected head.
    """
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    input_features = model.fc.in_features
    model.fc = nn.Linear(input_features, num_classes)
    if freeze_backbone:
        freeze_feature_extractor(model)
    return model


def build_efficientnet_b0(
    *,
    num_classes: int,
    pretrained: bool,
    freeze_backbone: bool,
) -> nn.Module:
    """Build an EfficientNet-B0 transfer-learning classifier.

    Args:
        num_classes: Number of output classes.
        pretrained: Whether to use ImageNet weights.
        freeze_backbone: Whether to freeze all layers except the final classifier.

    Returns:
        EfficientNet-B0 model with a binary-compatible classifier.
    """
    weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = efficientnet_b0(weights=weights)
    final_linear = model.classifier[1]
    if not isinstance(final_linear, nn.Linear):
        message = "The EfficientNet-B0 classifier is missing the expected Linear layer."
        raise TypeError(message)
    model.classifier[1] = nn.Linear(final_linear.in_features, num_classes)
    if freeze_backbone:
        freeze_feature_extractor(model)
    return model


def build_mobilenet_v3_small(
    *,
    num_classes: int,
    pretrained: bool,
    freeze_backbone: bool,
) -> nn.Module:
    """Build a MobileNetV3-Small transfer-learning classifier.

    Args:
        num_classes: Number of output classes.
        pretrained: Whether to use ImageNet weights.
        freeze_backbone: Whether to freeze all layers except the final classifier.

    Returns:
        MobileNetV3-Small model with a binary-compatible classifier.
    """
    weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = mobilenet_v3_small(weights=weights)
    final_linear = model.classifier[3]
    if not isinstance(final_linear, nn.Linear):
        message = "The MobileNetV3-Small classifier is missing the expected Linear layer."
        raise TypeError(message)
    model.classifier[3] = nn.Linear(final_linear.in_features, num_classes)
    if freeze_backbone:
        freeze_feature_extractor(model)
    return model


def freeze_feature_extractor(model: nn.Module) -> None:
    """Freeze backbone parameters while keeping classifier parameters trainable.

    Args:
        model: Supported transfer-learning model.

    Raises:
        ValueError: If the model architecture is unsupported.
    """
    for parameter in model.parameters():
        parameter.requires_grad = False

    classifier = get_classifier_module(model)
    for parameter in classifier.parameters():
        parameter.requires_grad = True


def unfreeze_fine_tuning_layers(model: nn.Module, model_name: str) -> None:
    """Unfreeze classifier and late feature layers for fine-tuning.

    Args:
        model: Supported transfer-learning model.
        model_name: Architecture name used to select the final feature block.

    Raises:
        ValueError: If the model architecture is unsupported.
    """
    normalized_model_name = model_name.strip().lower()
    for parameter in get_classifier_module(model).parameters():
        parameter.requires_grad = True

    if normalized_model_name == "resnet18":
        if not hasattr(model, "layer4"):
            message = "The ResNet18 model is missing its layer4 block."
            raise ValueError(message)
        for parameter in model.layer4.parameters():
            parameter.requires_grad = True
        return

    if normalized_model_name in {"efficientnet_b0", "efficientnet_v2_s", "mobilenet_v3_small"}:
        if not hasattr(model, "features"):
            message = f"The {model_name} model is missing its features block."
            raise ValueError(message)
        final_feature_block = model.features[-1]
        for parameter in final_feature_block.parameters():
            parameter.requires_grad = True
        return

    if normalized_model_name == "simple_cnn":
        for parameter in model.parameters():
            parameter.requires_grad = True
        return

    message = f"Unsupported model name for fine-tuning: {model_name}"
    raise ValueError(message)


def get_classifier_module(model: nn.Module) -> nn.Module:
    """Return the classifier module for a supported architecture.

    Args:
        model: Supported PyTorch classifier.

    Returns:
        Final classifier module.

    Raises:
        ValueError: If no supported classifier module is found.
    """
    if hasattr(model, "fc") and isinstance(model.fc, nn.Module):
        return model.fc
    if hasattr(model, "classifier") and isinstance(model.classifier, nn.Module):
        return model.classifier

    message = f"No supported classifier layer found: {model.__class__.__name__}"
    raise ValueError(message)


def count_trainable_parameters(model: nn.Module) -> int:
    """Count trainable model parameters.

    Args:
        model: PyTorch module.

    Returns:
        Number of parameters with `requires_grad=True`.
    """
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
