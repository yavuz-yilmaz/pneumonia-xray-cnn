"""Grad-CAM utilities for chest X-ray classifier explainability."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from torch import Tensor, nn


class GradCamError(RuntimeError):
    """Raised when Grad-CAM cannot be generated for a model or image."""


@dataclass(frozen=True)
class GradCamResult:
    """Grad-CAM output for one image.

    Attributes:
        class_id: Target class id used for the explanation.
        class_probability: Softmax probability for the explained class.
        heatmap: Normalized Grad-CAM heatmap resized to the original image size.
        overlay: RGB image array containing the original image with a heatmap overlay.
    """

    class_id: int
    class_probability: float
    heatmap: np.ndarray
    overlay: np.ndarray


class GradCam:
    """Generate Grad-CAM heatmaps for supported PyTorch image classifiers.

    Args:
        model: Trained PyTorch classifier in evaluation mode.
        target_layer: Final convolutional layer whose activations should be explained.
        device: Runtime device for forward and backward passes.

    Raises:
        GradCamError: If the target layer does not emit a tensor activation.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module, device: torch.device) -> None:
        self.model = model
        self.target_layer = target_layer
        self.device = device
        self.activations: Tensor | None = None
        self.gradients: Tensor | None = None
        self.forward_hook = self.target_layer.register_forward_hook(self._capture_activations)

    def close(self) -> None:
        """Remove registered model hooks."""
        self.forward_hook.remove()

    def generate(
        self,
        image_tensor: Tensor,
        original_image: Image.Image,
        *,
        target_class_id: int | None = None,
        overlay_alpha: float = 0.42,
    ) -> GradCamResult:
        """Generate a Grad-CAM heatmap and overlay for one image.

        Args:
            image_tensor: Normalized image tensor with shape `(3, height, width)`.
            original_image: Original PIL image used for overlay sizing and visualization.
            target_class_id: Optional class id to explain. If omitted, the predicted class is used.
            overlay_alpha: Heatmap opacity in the final overlay.

        Returns:
            Grad-CAM result with normalized heatmap and RGB overlay.

        Raises:
            GradCamError: If the input tensor shape is invalid or gradients cannot be computed.
            ValueError: If `overlay_alpha` is outside `[0.0, 1.0]`.
        """
        if image_tensor.ndim != 3:
            message = (
                "Grad-CAM input tensor shape `(3, H, W)` olmalı; "
                f"alınan shape: {tuple(image_tensor.shape)}"
            )
            raise GradCamError(message)
        if not 0.0 <= overlay_alpha <= 1.0:
            message = f"overlay_alpha [0.0, 1.0] aralığında olmalı; alınan değer: {overlay_alpha}"
            raise ValueError(message)

        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        self.activations = None
        self.gradients = None

        batch_tensor = image_tensor.unsqueeze(0).to(self.device)
        logits = self.model(batch_tensor)
        probabilities = torch.softmax(logits, dim=1)
        class_id = (
            int(torch.argmax(probabilities, dim=1).item())
            if target_class_id is None
            else target_class_id
        )

        if class_id < 0 or class_id >= logits.shape[1]:
            message = f"Geçersiz target_class_id={class_id}; model çıktı boyutu: {logits.shape[1]}"
            raise GradCamError(message)

        score = logits[:, class_id].sum()
        score.backward()

        if self.activations is None or self.gradients is None:
            message = "Grad-CAM için aktivasyon veya gradyan yakalanamadı."
            raise GradCamError(message)

        heatmap = self._build_heatmap(
            activations=self.activations.detach(),
            gradients=self.gradients.detach(),
            output_size=original_image.size,
        )
        overlay = create_heatmap_overlay(
            original_image=original_image,
            heatmap=heatmap,
            alpha=overlay_alpha,
        )
        return GradCamResult(
            class_id=class_id,
            class_probability=float(probabilities[0, class_id].detach().cpu().item()),
            heatmap=heatmap,
            overlay=overlay,
        )

    def _capture_activations(
        self,
        _module: nn.Module,
        _inputs: tuple[object, ...],
        output: Tensor,
    ) -> None:
        if not isinstance(output, Tensor):
            message = f"Grad-CAM hedef katmanı Tensor yerine {type(output)} döndürdü."
            raise GradCamError(message)
        self.activations = output
        output.register_hook(self._capture_gradients)

    def _capture_gradients(self, gradients: Tensor) -> None:
        self.gradients = gradients

    @staticmethod
    def _build_heatmap(
        *,
        activations: Tensor,
        gradients: Tensor,
        output_size: tuple[int, int],
    ) -> np.ndarray:
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        weighted_activations = (weights * activations).sum(dim=1, keepdim=True)
        class_activation_map = torch.relu(weighted_activations)
        resized_map = torch.nn.functional.interpolate(
            class_activation_map,
            size=(output_size[1], output_size[0]),
            mode="bilinear",
            align_corners=False,
        )
        heatmap = resized_map.squeeze().detach().cpu().numpy()
        minimum_value = float(np.min(heatmap))
        maximum_value = float(np.max(heatmap))
        if maximum_value <= minimum_value:
            return np.zeros_like(heatmap, dtype=np.float32)
        normalized_heatmap = (heatmap - minimum_value) / (maximum_value - minimum_value)
        return normalized_heatmap.astype(np.float32)


def resolve_target_layer(model: nn.Module, model_name: str) -> nn.Module:
    """Resolve the final convolutional feature layer for a supported model.

    Args:
        model: Supported classifier instance.
        model_name: Architecture name stored in the checkpoint or config.

    Returns:
        Module used as Grad-CAM target layer.

    Raises:
        GradCamError: If the model architecture is unsupported or malformed.
    """
    normalized_model_name = model_name.strip().lower()
    if normalized_model_name == "resnet18":
        if not hasattr(model, "layer4"):
            message = "ResNet18 modeli Grad-CAM için beklenen layer4 bloğunu içermiyor."
            raise GradCamError(message)
        return model.layer4[-1]

    if normalized_model_name == "simple_cnn":
        if not hasattr(model, "features"):
            message = "SimpleCNN modeli Grad-CAM için features bloğunu içermiyor."
            raise GradCamError(message)
        return model.features[-1]

    if normalized_model_name in {"efficientnet_b0", "mobilenet_v3_small"}:
        if not hasattr(model, "features"):
            message = f"{model_name} modeli Grad-CAM için features bloğunu içermiyor."
            raise GradCamError(message)
        return model.features[-1]

    message = f"Grad-CAM için desteklenmeyen model adı: {model_name}"
    raise GradCamError(message)


def create_heatmap_overlay(
    *,
    original_image: Image.Image,
    heatmap: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Blend a normalized heatmap over an original image.

    Args:
        original_image: Source image in any PIL mode.
        heatmap: Two-dimensional normalized heatmap in `[0.0, 1.0]`.
        alpha: Heatmap opacity in `[0.0, 1.0]`.

    Returns:
        RGB uint8 overlay array.

    Raises:
        ValueError: If the heatmap is not two-dimensional.
    """
    if heatmap.ndim != 2:
        message = f"Heatmap iki boyutlu olmalı; alınan shape: {heatmap.shape}"
        raise ValueError(message)

    original_array = np.asarray(original_image.convert("RGB"), dtype=np.float32) / 255.0
    colorized_heatmap = apply_jet_colormap(heatmap)
    blended = ((1.0 - alpha) * original_array) + (alpha * colorized_heatmap)
    return np.clip(blended * 255.0, 0, 255).astype(np.uint8)


def apply_jet_colormap(heatmap: np.ndarray) -> np.ndarray:
    """Apply matplotlib's Jet colormap to a normalized heatmap.

    Args:
        heatmap: Two-dimensional heatmap in `[0.0, 1.0]`.

    Returns:
        RGB float array in `[0.0, 1.0]`.
    """
    import matplotlib.pyplot as plt

    colormap = plt.get_cmap("jet")
    colorized = colormap(np.clip(heatmap, 0.0, 1.0))
    return colorized[..., :3].astype(np.float32)
