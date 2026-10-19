"""Optional DINOv3 classifier with a self-contained, offline checkpoint format."""

from __future__ import annotations

import os
from pathlib import Path

import torch
from torch import Tensor, nn


class DINOv3Classifier(nn.Module):
    """Classify concatenated CLS and mean patch features; exclude register tokens."""

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(2 * backbone.config.hidden_size, 2)

    @classmethod
    def from_pretrained(cls, directory: Path) -> DINOv3Classifier:
        """Load only a previously downloaded, revision-pinned local snapshot."""
        os.environ.setdefault("USE_TF", "0")
        os.environ.setdefault("USE_FLAX", "0")
        from transformers import AutoModel

        backbone = AutoModel.from_pretrained(
            directory,
            local_files_only=True,
            trust_remote_code=False,
            attn_implementation="sdpa",
            use_safetensors=True,
        )
        return cls(backbone)

    @classmethod
    def from_config(cls, config: dict) -> DINOv3Classifier:
        """Reconstruct an exported architecture without network access or weights."""
        os.environ.setdefault("USE_TF", "0")
        os.environ.setdefault("USE_FLAX", "0")
        from transformers import AutoConfig, AutoModel

        settings = dict(config)
        model_type = settings.pop("model_type")
        backbone_config = AutoConfig.for_model(model_type, **settings)
        return cls(AutoModel.from_config(backbone_config, attn_implementation="sdpa"))

    def forward(self, images: Tensor) -> Tensor:
        """Return NORMAL/PNEUMONIA logits using the complete image."""
        output = self.backbone(pixel_values=images)
        tokens = output.last_hidden_state
        patches = tokens[:, 1 + self.backbone.config.num_register_tokens :]
        features = torch.cat((tokens[:, 0], patches.mean(1)), dim=1)
        return self.classifier(features)

    def configure_training(self, tail_blocks: int) -> None:
        """Keep early layers frozen and in eval mode; optionally adapt the last blocks."""
        layers = self.backbone.layer
        if not 0 <= tail_blocks <= len(layers):
            raise ValueError("Invalid number of DINOv3 tail blocks")
        self.backbone.requires_grad_(False)
        self.backbone.eval()
        self.classifier.requires_grad_(True)
        self.classifier.train()
        if tail_blocks:
            for block in layers[-tail_blocks:]:
                block.requires_grad_(True)
                block.train()
            self.backbone.norm.requires_grad_(True)
            self.backbone.norm.train()
