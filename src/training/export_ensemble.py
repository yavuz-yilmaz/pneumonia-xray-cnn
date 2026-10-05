"""Package exactly three completed, compatible ResNet18 runs without fitting weights."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from src.training.models import ResNet18Ensemble
from src.training.train_clean import verify_manifests


def export(members: list[Path], output: Path, manifests: Path) -> None:
    """Preserve member provenance and reject mismatched splits or preprocessing."""
    if len(members) != 3 or len(set(members)) != 3:
        raise ValueError("Exactly three distinct member directories are required")
    if output.exists():
        raise FileExistsError("Ensemble output already exists")
    audit = verify_manifests(manifests)
    ensemble = ResNet18Ensemble()
    metadata = None
    provenance = []
    for directory, model in zip(members, ensemble.members, strict=True):
        if json.loads((directory / "result.json").read_bytes())["status"] != "complete":
            raise ValueError("Member training is incomplete")
        path = directory / "best_model.pt"
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        protocol = checkpoint["training_protocol"]
        if checkpoint["model_name"] != "resnet18":
            raise ValueError("Ensemble requires ResNet18 members")
        if protocol["data_audit"]["manifest_sha256"] != audit["manifest_sha256"]:
            raise ValueError("Members use different data splits")
        current = {
            key: checkpoint[key]
            for key in ("image_size", "normalization", "preprocessing", "label_mapping")
        }
        if metadata is not None and current != metadata:
            raise ValueError("Member preprocessing or labels differ")
        metadata = current
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        provenance.append(
            {
                "checkpoint": path.as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "seed": protocol["seed"],
            }
        )
    protocol = {
        "data_audit": audit,
        "minimum_recall": 0.98,
        "members": provenance,
        "combination": "Equal probability average; no learned ensemble weights",
        "test_access": "none",
    }
    output.mkdir(parents=True)
    torch.save(
        dict(
            metadata,
            model_name="resnet18_ensemble3",
            model_version=output.name,
            model_state_dict=ensemble.state_dict(),
            decision_threshold=0.5,
            training_protocol=protocol,
        ),
        output / "best_model.pt",
    )
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    (output / "result.json").write_text(
        json.dumps({"status": "complete", "requires_validation_threshold_selection": True}),
        encoding="utf-8",
    )


def main() -> None:
    """Export before the separate validation-only selection command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--members", type=Path, nargs=3, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifests", type=Path, default=Path("data/processed/clean_v1"))
    args = parser.parse_args()
    export(args.members, args.output, args.manifests)


if __name__ == "__main__":
    main()
