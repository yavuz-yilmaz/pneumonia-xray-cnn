"""Evaluate fixed pneumonia scores against an independently reserved opacity cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from src.evaluation.decision import binary_metrics
from src.evaluation.evaluate_clean import bootstrap_intervals
from src.inference.predict import load_model, preprocess_image


def verified_images(cohort: Path) -> list[dict]:
    """Require the locked cohort, successful overlap audit, and unchanged images."""
    protocol = json.loads((cohort / "protocol.json").read_bytes())
    audit = json.loads((cohort / "image_audit.json").read_bytes())
    if audit["overlaps"]:
        raise ValueError("External audit reports overlaps")
    if (
        hashlib.sha256((cohort / "cohort.json").read_bytes()).hexdigest()
        != protocol["cohort_sha256"]
    ):
        raise ValueError("External cohort changed")
    if audit["cohort_sha256"] != protocol["cohort_sha256"]:
        raise ValueError("Image audit belongs to another cohort")
    content = (cohort / "images.json").read_bytes()
    if hashlib.sha256(content).hexdigest() != audit["images_manifest_sha256"]:
        raise ValueError("External image manifest changed")
    records = json.loads(content)
    locked_records = json.loads((cohort / "cohort.json").read_bytes())
    locked_patients = {r["nih_patient_id"]: r for r in locked_records}
    if len(locked_patients) != len(locked_records) or len(records) != len(locked_records):
        raise ValueError("Extracted images do not match the locked patient cohort")
    if len(records) != audit["images"] or len({r["nih_patient_id"] for r in records}) != len(
        records
    ):
        raise ValueError("External patient count mismatch")
    for record in records:
        locked = locked_patients.get(record["nih_patient_id"])
        if locked is None or any(record.get(key) != value for key, value in locked.items()):
            raise ValueError("Extracted image identity or label changed from the locked cohort")
        if record["label_id"] not in (0, 1):
            raise ValueError("Invalid external class identifier")
        expected_label = ("Normal", "Lung Opacity")[record["label_id"]]
        if record["source_label"] != expected_label:
            raise ValueError("External source label mismatch")
        if hashlib.sha256(Path(record["filepath"]).read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError("External image changed after audit")
    return records


def evaluate(selected: Path, baseline: Path, cohort: Path, output: Path) -> dict:
    """Compare the validation-locked winner and historical baseline without tuning."""
    if output.exists():
        raise FileExistsError("External evaluation already started; refusing overwrite")
    selection = json.loads((selected / "selection.json").read_bytes())
    winner = selected / "best_model.pt"
    hashes = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in {"baseline": baseline, "selected": winner}.items()
    }
    if hashes["selected"] != selection["selected_checkpoint_sha256"]:
        raise ValueError("Selected checkpoint changed")
    records = verified_images(cohort)
    labels = np.array([r["label_id"] for r in records])
    groups = np.array([r["nih_patient_id"] for r in records])
    # Pin both inputs before any model scores are computed on the external cohort.
    output.mkdir(parents=True)
    lock = {
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_sha256": hashes,
        "image_audit": json.loads((cohort / "image_audit.json").read_bytes()),
        "threshold_policy": "Use each checkpoint's existing threshold; no external calibration",
    }
    (output / "inputs.json").write_text(json.dumps(lock, indent=2), encoding="utf-8")
    result = {"target": "RSNA lung opacity versus normal; not confirmed pneumonia", "results": {}}
    for name, path in (("baseline", baseline), ("selected", winner)):
        loaded = load_model(path)
        if loaded.model_name == "xrv_densenet121":
            raise ValueError("XRV NIH/RSNA pretraining invalidates independent evaluation")
        scores = []
        with torch.inference_mode():
            for start in range(0, len(records), 8):
                images = torch.cat(
                    [
                        preprocess_image(
                            Path(r["filepath"]),
                            loaded.image_size,
                            loaded.normalization_mean,
                            loaded.normalization_std,
                            preprocessing=loaded.preprocessing,
                        )
                        for r in records[start : start + 8]
                    ]
                )
                scores.extend(
                    loaded.model(images.to(loaded.device)).softmax(1)[:, 1].cpu().tolist()
                )
        probabilities = np.array(scores)
        np.savez(
            output / f"{name}_predictions.npz",
            labels=labels,
            probabilities=probabilities,
            patient_ids=groups,
        )
        metrics = binary_metrics(labels, probabilities, loaded.decision_threshold)
        result["results"][name] = {
            "checkpoint_sha256": hashes[name],
            "metrics": metrics,
            "confidence_intervals_95": bootstrap_intervals(
                labels, probabilities, loaded.decision_threshold, groups
            ),
        }
        print(json.dumps({"model": name, "metrics": metrics}), flush=True)
        del loaded
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    result["limitations"] = json.loads((cohort / "protocol.json").read_bytes())["limitations"]
    (output / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    """Run only after pediatric model selection has completed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected", type=Path, default=Path("models/clean_v4/selected"))
    parser.add_argument("--baseline", type=Path, default=Path("models/best_model.pt"))
    parser.add_argument("--cohort", type=Path, default=Path("data/processed/rsna_external_v1"))
    parser.add_argument("--output", type=Path, default=Path("reports/metrics/rsna_external_v1"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    evaluate(args.selected, args.baseline, args.cohort, args.output)


if __name__ == "__main__":
    main()
