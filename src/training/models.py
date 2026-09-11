"""Model definitions for chest X-ray pneumonia classification."""

from __future__ import annotations

from torch import Tensor, nn
from torchvision.models import (
    EfficientNet_B0_Weights,
    MobileNet_V3_Small_Weights,
    ResNet18_Weights,
    efficientnet_b0,
    mobilenet_v3_small,
    resnet18,
)


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
            message = f"num_classes en az 2 olmalı; alınan değer: {num_classes}"
            raise ValueError(message)
        if not 0.0 <= dropout_probability < 1.0:
            message = (
                "dropout_probability [0.0, 1.0) aralığında olmalı; "
                f"alınan değer: {dropout_probability}"
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
        "Desteklenen model adları: simple_cnn, resnet18, efficientnet_b0, "
        f"mobilenet_v3_small. Alınan değer: {model_name}"
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
        message = "EfficientNet-B0 classifier beklenen Linear katmanı içermiyor."
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
        message = "MobileNetV3-Small classifier beklenen Linear katmanı içermiyor."
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
            message = "ResNet18 modeli layer4 bloğunu içermiyor."
            raise ValueError(message)
        for parameter in model.layer4.parameters():
            parameter.requires_grad = True
        return

    if normalized_model_name in {"efficientnet_b0", "mobilenet_v3_small"}:
        if not hasattr(model, "features"):
            message = f"{model_name} modeli features bloğunu içermiyor."
            raise ValueError(message)
        final_feature_block = model.features[-1]
        for parameter in final_feature_block.parameters():
            parameter.requires_grad = True
        return

    if normalized_model_name == "simple_cnn":
        for parameter in model.parameters():
            parameter.requires_grad = True
        return

    message = f"Fine-tuning için desteklenmeyen model adı: {model_name}"
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

    message = f"Desteklenen classifier katmanı bulunamadı: {model.__class__.__name__}"
    raise ValueError(message)


def count_trainable_parameters(model: nn.Module) -> int:
    """Count trainable model parameters.

    Args:
        model: PyTorch module.

    Returns:
        Number of parameters with `requires_grad=True`.
    """
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
