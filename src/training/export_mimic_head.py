"""Export the author's MIMIC pneumonia head with a source-validation threshold."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from src.data.download_nih import file_sha256
from src.data.nih_dataset import NIH_LABEL_TO_ID
from src.evaluation.decision import binary_metrics, select_f1_threshold
from src.training.train_xray_probe import (
    MIMIC_FEATURE_SHA256,
    MIMIC_SOURCE_SHA256,
    XRAY_MEAN,
    XRAY_STD,
    load_pretrained_features,
)

HEAD_SHA256 = "2381ed671a42a3c02418752b332660c37ca6c1f8ae686f715e63a9e67073b94e"


def load_source_head(path: Path) -> dict:
    """Require the exact safe tensor artifact and pneumonia label provenance."""
    if file_sha256(path) != HEAD_SHA256:
        raise ValueError("Source head differs from the reviewed artifact")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    metadata = saved["metadata"]
    if (
        metadata.get("weights") != "densenet121-res224-mimic_ch"
        or metadata.get("source_sha256") != MIMIC_SOURCE_SHA256
        or metadata.get("feature_checkpoint_sha256") != MIMIC_FEATURE_SHA256
        or metadata.get("pathology") != "Pneumonia"
        or metadata.get("source_row") != 8
        or metadata.get("publisher_supported_label") != "Pneumonia"
        or metadata.get("source_targets", [])[8:9] != ["Pneumonia"]
    ):
        raise ValueError("Source head lacks verified MIMIC pneumonia provenance")
    if (
        saved["weight"].shape != (1024,)
        or saved["bias"].ndim != 0
        or not torch.isfinite(saved["weight"]).all()
        or not torch.isfinite(saved["bias"])
    ):
        raise ValueError("Invalid source classifier parameters")
    return saved


def export(root: Path, output: Path) -> dict:
    """Reuse development features only; keep the externally trained head fixed."""
    if output.exists():
        raise FileExistsError("Source-head experiment already exists; preserving it")
    run = root / "reports/metrics/nih_mimic_run_v1"
    status = json.loads((run / "status.json").read_bytes())
    if status["state"] != "complete_train_validation_only":
        raise ValueError("Registered MIMIC feature probe must finish first")
    registration = json.loads((run / "registration.json").read_bytes())
    for name, digest in registration["source_sha256"].items():
        if file_sha256(root / name) != digest:
            raise ValueError("Registered development recipe/data changed")
    directory = root / "models/nih_mimic_v1/frozen"
    result = json.loads((directory / "result.json").read_bytes())
    if result["status"] != "complete":
        raise ValueError("Frozen development features are incomplete")
    head_path = root / "models/pretrained/xrv_mimic_ch/pneumonia_head_safe.pt"
    source = load_source_head(head_path)
    feature_path = root / "models/pretrained/xrv_mimic_ch/features_safe.pt"
    model = load_pretrained_features("mimic_ch", feature_path).cpu().eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    with torch.no_grad():
        model.classifier.weight.zero_()
        model.classifier.bias.zero_()
        model.classifier.weight[1].copy_(source["weight"])
        model.classifier.bias[1].copy_(source["bias"])
    manifest = root / "data/processed/nih_pneumonia_v1/val_manifest.csv"
    with manifest.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    # np.load is lazy: neither train features nor train labels are opened.
    feature_file = directory / "features.npz"
    with np.load(feature_file, allow_pickle=False) as saved:
        features, labels = saved["val"], saved["val_labels"]
    if (
        features.shape != (len(rows), 1024)
        or not np.isfinite(features).all()
        or not np.array_equal(labels, [int(r["label_id"]) for r in rows])
        or {int(v) for v in labels} != {0, 1}
    ):
        raise ValueError("Validation feature/label alignment failed")
    torch.set_num_threads(4)
    with torch.inference_mode():
        logits = model.classifier(torch.from_numpy(features))
        probabilities = logits.softmax(1)[:, 1].numpy()
    threshold = select_f1_threshold(labels, probabilities, minimum_recall=0.9)
    metrics = binary_metrics(labels, probabilities, threshold)
    protocol = {
        "task": "nih_report_pneumonia",
        "pretraining": "mimic_ch",
        "backbone_training": "frozen",
        "head_training_on_nih": "none",
        "head_origin": "Original MIMIC pneumonia classifier; source row 8",
        "head_artifact_sha256": HEAD_SHA256,
        "minimum_recall": 0.9,
        "threshold_rule": "Validation F1 under recall >=0.90",
        "source_metadata": source["metadata"],
        "external_weights_sha256": MIMIC_SOURCE_SHA256,
        "pretraining_metadata": model.pretraining_metadata,
        "test_access": "none",
        "openi_access": "none",
        "feature_scaler": "none",
        "validation_manifest_sha256": file_sha256(manifest),
        "development_features_sha256": file_sha256(feature_file),
        "prerequisite_registration_sha256": file_sha256(run / "registration.json"),
        "source_sha256": file_sha256(Path(__file__)),
        "scope": "Secondary fixed source-head transfer; does not replace primary C-selected head",
    }
    checkpoint = {
        "model_name": "xrv_densenet121",
        "image_size": 224,
        "normalization": {"mean": XRAY_MEAN, "std": XRAY_STD},
        "preprocessing": "resize",
        "label_mapping": NIH_LABEL_TO_ID,
        "decision_threshold": threshold,
        "validation_metrics": metrics,
        "model_version": "nih_mimic_original_pneumonia_head",
        "model_state_dict": model.state_dict(),
        "training_protocol": protocol,
    }
    output.mkdir(parents=True)
    torch.save(checkpoint, output / "best_model.pt")
    np.savez(output / "validation_predictions.npz", labels=labels, probabilities=probabilities)
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    summary = {
        "status": "complete",
        "validation": metrics,
        "checkpoint_sha256": file_sha256(output / "best_model.pt"),
        "test_access": "none",
        "openi_access": "none",
        "not_a_new_nih_fit": True,
        "external_evaluation_required": True,
    }
    (output / "result.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    """Calibrate the fixed original source head after development extraction."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("models/nih_mimic_v1/source_head"))
    args = parser.parse_args()
    root = args.root.resolve()
    print(json.dumps(export(root, root / args.output), indent=2), flush=True)


if __name__ == "__main__":
    main()
