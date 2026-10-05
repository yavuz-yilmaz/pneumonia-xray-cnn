"""Compare fixed source-selected models on a locked, independent OpenI cohort."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import LABEL_TO_ID, ChestXRayDataset
from src.data.download_nih import file_sha256
from src.data.image_preparation import CheckpointImageTransform
from src.data.nih_dataset import NIH_LABEL_TO_ID
from src.evaluation.decision import binary_metrics
from src.training.models import build_model
from src.training.train_xray_probe import MIMIC_FEATURE_SHA256, MIMIC_SOURCE_SHA256

LABELS = {"NO_CODED_OR_MENTIONED_PNEUMONIA": 0, "MANUAL_REPORT_PNEUMONIA": 1}
METRICS = (
    "accuracy",
    "precision",
    "recall",
    "f1",
    "specificity",
    "balanced_accuracy",
    "roc_auc",
    "average_precision",
)
MODEL_NAMES = {"mimic", "nih", "original", "current"}
BASELINE_HASHES = {
    "original": "6dd50d60330b52664236c74a45696a04770723e91c5dbea3715e322709a995d7",
    "current": "da5c58d36865cc01d8df1dce5c358dac2a06b0b0b4981115bb7e35d806e2be9b",
}
SOURCE_FILES = (
    "src/evaluation/evaluate_openi.py",
    "src/evaluation/decision.py",
    "src/data/image_preparation.py",
    "src/data/dataset.py",
    "src/data/nih_dataset.py",
    "src/data/transforms.py",
    "src/data/download_nih.py",
    "src/training/models.py",
    "src/training/train_xray_probe.py",
    "docs/openi_external_protocol.md",
)


class OpenIReportDataset(ChestXRayDataset):
    """Keep manual report-code labels distinct from clinical normality."""

    label_mapping = LABELS


def verify_hashes(root: Path, hashes: dict[str, str]) -> None:
    """Refuse drift in every explicitly pinned artifact before inference."""
    for name, digest in hashes.items():
        if file_sha256(root / name) != digest:
            raise ValueError(f"Registered artifact changed: {name}")


def derive_cohort(
    candidates: list[dict], images: list[dict], matches: list[dict]
) -> tuple[list[dict], list[dict]]:
    """Remove whole overlapping cases, then keep unique frontal decoded images."""
    by_image = {r["image_id"]: r for r in images}
    by_case = {r["case_id"]: r for r in candidates}
    if len(by_image) != len(images) or len(by_case) != len(candidates):
        raise ValueError("Repeated source image/case identifier")
    blocked: dict[str, set[str]] = defaultdict(set)
    for match in matches:
        image = by_image[match["openi_image_id"]]
        if image["case_id"] != match["openi_case_id"]:
            raise ValueError("Duplicate match disagrees with image ownership")
        case = image["case_id"]
        if match["reference_source"] in {"NIH", "legacy"}:
            blocked[case].add("cross_source_duplicate")
        elif match["reference_source"] == "OpenI":
            other = by_image[match["reference_image_id"]]
            if other["case_id"] != match["reference_case_id"]:
                raise ValueError("Duplicate reference ownership differs")
            if case != other["case_id"]:
                blocked[case].add("cross_case_duplicate")
                blocked[other["case_id"]].add("cross_case_duplicate")
        else:
            raise ValueError("Unknown duplicate reference source")
    retained, excluded = [], []
    for case in sorted(candidates, key=lambda r: r["case_id"]):
        case_id = case["case_id"]
        reasons = set(case["exclusion_reasons"]) | blocked[case_id]
        if not case["metadata_eligible"]:
            reasons.add("metadata_ineligible")
        if any(image not in by_image for image in case["image_ids"]):
            reasons.add("missing_linked_image")
        available = [by_image[i] for i in case["image_ids"] if i in by_image]
        if any(r["case_id"] != case_id for r in available):
            raise ValueError("Candidate image linked to a different case")
        frontal = sorted(
            (r for r in available if r["secondary_projection"] == "Frontal"),
            key=lambda r: r["image_id"],
        )
        unique = {}
        for image in frontal:
            unique.setdefault(image["pixel_sha256"], image)
        if not unique:
            reasons.add("no_mapped_frontal_image")
        if reasons:
            excluded.append({"case_id": case_id, "reasons": sorted(reasons)})
            continue
        label = case["candidate_label"]
        if label not in LABELS:
            raise ValueError("Unknown report-code target")
        retained.append(
            {
                "case_id": case_id,
                "label": label,
                "label_id": LABELS[label],
                "images": list(unique.values()),
            }
        )
    if {r["label_id"] for r in retained} != {0, 1}:
        raise ValueError("External cohort must retain both classes")
    return retained, excluded


def prepare(root: Path, output: Path) -> dict:
    """Lock the auditable cohort without reading any checkpoint or model score."""
    if output.exists():
        raise FileExistsError("OpenI cohort already exists; preserving it")
    evidence = root / "reports/data_access_20261001"
    recipe = json.loads((evidence / "openi_external_recipe_registration.json").read_bytes())
    if recipe["initial_nih_test_started"] or not recipe["no_openi_model_inference"]:
        raise ValueError("Prospective recipe was not registered before scoring")
    verify_hashes(root, recipe["recipe_and_available_input_sha256"])
    directory = root / "data/processed/openi_images_v2"
    status = json.loads((directory / "status.json").read_bytes())
    audit = json.loads((directory / "image_audit.json").read_bytes())
    if (
        status["state"] != "complete_image_audit_only"
        or audit["gzip_crc_verified"] is not True
        or audit["all_archive_image_pixels_decoded"] is not True
        or audit["no_model_inference"] is not True
        or audit["reference_counts"]["NIH"] != 112120
        or set(audit["artifact_sha256"])
        != {"images.json", "thumbnails.npz", "duplicate_matches.json"}
    ):
        raise ValueError("OpenI source/overlap audit incomplete")
    verify_hashes(directory, audit["artifact_sha256"])
    registration = json.loads((directory / "registration.json").read_bytes())
    verify_hashes(root, registration["input_sha256"])
    candidate_path = root / "data/processed/openi_candidate_v1/metadata_candidates.json"
    images = json.loads((directory / "images.json").read_bytes())
    if len(images) != audit["images"]:
        raise ValueError("Image audit count differs from decoded inventory")
    matches = json.loads((directory / "duplicate_matches.json").read_bytes())
    projection_path = evidence / "openi_secondary_projection_mapping.json"
    projection_rows = json.loads(projection_path.read_bytes())
    projections = {r["image_id"]: r for r in projection_rows}
    if len(projections) != len(projection_rows):
        raise ValueError("Repeated secondary view mapping")
    for image in images:
        mapping = projections.get(image["image_id"])
        if image["secondary_projection"] != (mapping["projection"] if mapping else None):
            raise ValueError("Image view differs from registered secondary mapping")
        if mapping and image["case_id"] != mapping["case_id"]:
            raise ValueError("View mapping differs from official case linkage")
    cases, exclusions = derive_cohort(json.loads(candidate_path.read_bytes()), images, matches)
    rows = []
    for case in cases:
        for image in case["images"]:
            if file_sha256(root / image["filepath"]) != image["sha256"]:
                raise ValueError("Retained image changed since decoding audit")
            rows.append(
                {
                    "filepath": (root / image["filepath"]).as_posix(),
                    "split": "external",
                    "label": case["label"],
                    "label_id": case["label_id"],
                    "case_id": case["case_id"],
                    "image_id": image["image_id"],
                    "sha256": image["sha256"],
                    "pixel_sha256": image["pixel_sha256"],
                }
            )
    output.mkdir(parents=True)
    (output / "cases.json").write_text(json.dumps(cases, indent=2), encoding="utf-8")
    (output / "exclusions.json").write_text(json.dumps(exclusions, indent=2), encoding="utf-8")
    with (output / "manifest.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    locked = {
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "case_count": len(cases),
        "image_count": len(rows),
        "case_labels": dict(Counter(c["label"] for c in cases)),
        "excluded_cases": len(exclusions),
        "no_model_inference": True,
        "unit": "case; mean probability over unique frontal images",
        "target": "manual report pneumonia code; absence of code/mention is not clinical normality",
        "artifact_sha256": {
            n: file_sha256(output / n) for n in ("cases.json", "exclusions.json", "manifest.csv")
        },
        "input_sha256": registration["input_sha256"],
        "image_audit_sha256": file_sha256(directory / "image_audit.json"),
        "source_sha256": {name: file_sha256(root / name) for name in SOURCE_FILES},
    }
    (output / "cohort_audit.json").write_text(json.dumps(locked, indent=2), encoding="utf-8")
    return locked


def aggregate_cases(
    rows: list[dict], labels: np.ndarray, probabilities: np.ndarray
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Use equal case weight despite differing numbers of frontal images."""
    if len(rows) != len(labels) or probabilities.shape != labels.shape:
        raise ValueError("Image prediction alignment mismatch")
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Invalid image probabilities")
    grouped = defaultdict(list)
    targets = {}
    for row, target, score in zip(rows, labels, probabilities, strict=True):
        if int(row["label_id"]) != target:
            raise ValueError("Manifest and loader target differ")
        case = row["case_id"]
        if case in targets and targets[case] != target:
            raise ValueError("Conflicting case labels")
        targets[case] = int(target)
        grouped[case].append(float(score))
    ids = sorted(grouped)
    return ids, np.array([targets[c] for c in ids]), np.array([np.mean(grouped[c]) for c in ids])


def case_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    """Mark undefined precision explicitly instead of inventing positive evidence."""
    metrics = binary_metrics(labels, probabilities, threshold)
    metrics["undefined_metrics"] = []
    if not (probabilities >= threshold).any():
        metrics["precision"] = None
        metrics["undefined_metrics"].append("precision")
    return metrics


def paired_intervals(
    labels: np.ndarray, scores: dict, thresholds: dict, repetitions: int = 1000
) -> dict:
    """Resample the same case positions for all models and paired differences."""
    if set(scores) != MODEL_NAMES or set(thresholds) != MODEL_NAMES or repetitions < 1:
        raise ValueError("All four fixed models are required for paired intervals")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Both classes are required")
    for name, values in scores.items():
        if (
            values.shape != labels.shape
            or not np.isfinite(values).all()
            or ((values < 0) | (values > 1)).any()
        ):
            raise ValueError("Invalid paired predictions")
        if not np.isfinite(thresholds[name]) or not 0 <= thresholds[name] <= 1:
            raise ValueError("Invalid fixed threshold")
    values = {name: {metric: [] for metric in METRICS} for name in MODEL_NAMES}
    differences = {name: {metric: [] for metric in METRICS} for name in MODEL_NAMES - {"mimic"}}
    rng = np.random.default_rng(20261001)
    valid = 0
    for _ in range(repetitions):
        positions = rng.choice(len(labels), len(labels), replace=True)
        if len(np.unique(labels[positions])) < 2:
            continue
        valid += 1
        results = {
            name: case_metrics(labels[positions], scores[name][positions], thresholds[name])
            for name in MODEL_NAMES
        }
        for name in MODEL_NAMES:
            for metric in METRICS:
                result = results[name][metric]
                if result is not None:
                    values[name][metric].append(result)
                if name != "mimic" and result is not None and results["mimic"][metric] is not None:
                    differences[name][metric].append(results["mimic"][metric] - result)

    def summarize(items: dict) -> dict:
        return {
            name: {
                metric: {
                    "valid_repetitions": len(samples),
                    "interval_95": np.percentile(samples, [2.5, 97.5]).tolist()
                    if samples
                    else None,
                }
                for metric, samples in per_metric.items()
            }
            for name, per_metric in items.items()
        }

    return {
        "seed": 20261001,
        "requested_repetitions": repetitions,
        "both_class_repetitions": valid,
        "models": summarize(values),
        "mimic_minus_comparator": summarize(differences),
    }


def model_inputs(root: Path) -> dict:
    """Require the previously registered source selection and MIMIC provenance."""
    nih_selection = json.loads((root / "models/nih_v1/selected/selection.json").read_bytes())
    nih_status = json.loads((root / "reports/metrics/nih_run_v1/status.json").read_bytes())
    mimic_status = json.loads((root / "reports/metrics/nih_mimic_run_v1/status.json").read_bytes())
    if (
        nih_status["state"] != "complete"
        or mimic_status["state"] != "complete_train_validation_only"
    ):
        raise ValueError("Both registered training chains must complete before external scoring")
    probe_registration = json.loads(
        (root / "reports/metrics/nih_mimic_run_v1/registration.json").read_bytes()
    )
    if probe_registration["prerequisite_test_started_at_registration"] is not False:
        raise ValueError("MIMIC probe was not prospectively registered")
    verify_hashes(root, probe_registration["source_sha256"])
    result = json.loads((root / "models/nih_mimic_v1/frozen/result.json").read_bytes())
    winner = max(
        result["candidates"], key=lambda c: (c["validation"]["average_precision"], -c["C"])
    )
    if (
        result["status"] != "complete"
        or result["selected_candidate"] != winner
        or len(result["candidates"]) != 4
        or {c["C"] for c in result["candidates"]} != {0.001, 0.01, 0.1, 1.0}
    ):
        raise ValueError("MIMIC candidate does not match fixed validation selection")
    paths = {
        "mimic": root / winner["directory"] / "best_model.pt",
        "nih": root / "models/nih_v1/selected/best_model.pt",
        "current": root / "models/clean_v3/selected_matched98/best_model.pt",
        "original": root / "models/best_model.pt",
    }
    if paths["mimic"].resolve().parent.parent != (root / "models/nih_mimic_v1/frozen").resolve():
        raise ValueError("MIMIC candidate lies outside the registered experiment")
    for name, expected in dict(
        BASELINE_HASHES, nih=nih_selection["selected_checkpoint_sha256"]
    ).items():
        if file_sha256(paths[name]) != expected:
            raise ValueError("Fixed source comparator changed")
    if nih_selection["test_used_for_selection"] is not False:
        raise ValueError("NIH selection accessed test")
    protocol = json.loads(paths["mimic"].with_name("protocol.json").read_bytes())
    metadata = protocol.get("pretraining_metadata", {})
    if (
        protocol.get("pretraining") != "mimic_ch"
        or protocol.get("test_access") != "none"
        or protocol.get("backbone_training") != "frozen"
        or protocol.get("external_weights_sha256") != MIMIC_SOURCE_SHA256
        or metadata.get("weights") != "densenet121-res224-mimic_ch"
        or metadata.get("feature_checkpoint_sha256") != MIMIC_FEATURE_SHA256
        or metadata.get("source_sha256") != MIMIC_SOURCE_SHA256
        or protocol.get("task") != "nih_report_pneumonia"
        or protocol.get("C") != winner["C"]
        or protocol.get("minimum_recall") != 0.9
        or protocol.get("source_sha256")
        != probe_registration["source_sha256"]["src/training/train_xray_probe.py"]
        or protocol.get("data_audit", {}).get("manifest_sha256")
        != nih_selection["data_audit"]["manifest_sha256"]
    ):
        raise ValueError("Mixed/unverified pretraining cannot score OpenI")
    records = {
        name: {"path": str(path), "sha256": file_sha256(path)} for name, path in paths.items()
    }
    records["mimic"]["expected_validation"] = winner["validation"]
    return records


def read_model(name: str, record: dict) -> dict:
    """Read only hash-pinned local source checkpoints before recording metadata."""
    path = Path(record["path"])
    if file_sha256(path) != record["sha256"]:
        raise ValueError("Checkpoint changed before loading")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    mapping = NIH_LABEL_TO_ID if name in {"mimic", "nih"} else LABEL_TO_ID
    if checkpoint.get("label_mapping") != mapping:
        raise ValueError("Checkpoint task differs from registered source")
    if name == "mimic":
        protocol = json.loads(path.with_name("protocol.json").read_bytes())
        if (
            checkpoint["model_name"] != "xrv_densenet121"
            or json.dumps(checkpoint["training_protocol"], sort_keys=True)
            != json.dumps(protocol, sort_keys=True)
            or checkpoint["validation_metrics"] != record["expected_validation"]
        ):
            raise ValueError("MIMIC checkpoint provenance differs from registered protocol")
    elif checkpoint["model_name"] == "xrv_densenet121":
        raise ValueError("Unverified XRV source cannot score OpenI")
    threshold = float(checkpoint.get("decision_threshold", 0.70))
    if (
        not np.isfinite(threshold)
        or not 0 <= threshold <= 1
        or (name == "original" and threshold != 0.70)
    ):
        raise ValueError("Checkpoint threshold differs from fixed source policy")
    checkpoint["decision_threshold"] = threshold
    transform = CheckpointImageTransform.from_metadata(checkpoint)
    if (
        not 16 <= transform.image_size <= 1024
        or transform.preprocessing not in {"resize", "clahe"}
        or len(transform.normalization_mean) != 3
        or len(transform.normalization_std) != 3
        or not np.isfinite(transform.normalization_mean).all()
        or not np.isfinite(transform.normalization_std).all()
        or (np.asarray(transform.normalization_std) <= 0).any()
    ):
        raise ValueError("Invalid checkpoint input contract")
    return checkpoint


def predict_images(checkpoint: dict, manifest: Path, device: torch.device) -> tuple:
    """Run full-precision image inference using the stored input contract."""
    dataset = OpenIReportDataset(manifest, CheckpointImageTransform.from_metadata(checkpoint))
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
    model = build_model(checkpoint["model_name"], 2, pretrained=False, freeze_backbone=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    labels, scores = [], []
    with torch.inference_mode():
        for images, targets in loader:
            logits = model(images.to(device))
            if logits.shape != (len(targets), 2) or not torch.isfinite(logits).all():
                raise ValueError("Invalid model logits")
            labels.extend(targets.tolist())
            scores.extend(logits.softmax(1)[:, 1].cpu().tolist())
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return np.asarray(labels), np.asarray(scores)


def evaluate(root: Path, cohort: Path, output: Path, device_name: str = "auto") -> dict:
    """Score the four-model locked comparison once, preserving failed runs."""
    if output.exists():
        raise FileExistsError("External evaluation already registered; preserving it")
    audit = json.loads((cohort / "cohort_audit.json").read_bytes())
    verify_hashes(cohort, audit["artifact_sha256"])
    verify_hashes(root, audit["input_sha256"])
    verify_hashes(root, audit["source_sha256"])
    with (cohort / "manifest.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        if file_sha256(Path(row["filepath"])) != row["sha256"]:
            raise ValueError("Cohort image changed before inference")
    inputs = model_inputs(root)
    checkpoints = {name: read_model(name, record) for name, record in inputs.items()}
    output.mkdir(parents=True)
    thresholds = {name: c["decision_threshold"] for name, c in checkpoints.items()}
    registration = {
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "models": inputs,
        "thresholds": thresholds,
        "cohort_audit_sha256": file_sha256(cohort / "cohort_audit.json"),
        "source_sha256": audit["source_sha256"],
        "cohort": audit,
        "input_contracts": {
            name: {
                key: c.get(key)
                for key in ("image_size", "normalization", "preprocessing", "label_mapping")
            }
            for name, c in checkpoints.items()
        },
        "test_used_for_selection": False,
        "no_openi_threshold_search": True,
    }
    (output / "registration.json").write_text(json.dumps(registration, indent=2), encoding="utf-8")
    status = {"state": "started", "completed_models": []}

    def save_status() -> None:
        temporary = output / "status.tmp"
        temporary.write_text(json.dumps(status, indent=2), encoding="utf-8")
        temporary.replace(output / "status.json")

    save_status()
    try:
        torch.set_num_threads(4)
        device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device_name == "auto"
            else torch.device(device_name)
        )
        results, scores, case_ids, labels = {}, {}, None, None
        for name in ("mimic", "nih", "current", "original"):
            image_labels, image_scores = predict_images(
                checkpoints[name], cohort / "manifest.csv", device
            )
            ids, targets, probabilities = aggregate_cases(rows, image_labels, image_scores)
            if case_ids is not None and (ids != case_ids or not np.array_equal(labels, targets)):
                raise ValueError("Models were not evaluated on identical cases")
            case_ids, labels = ids, targets
            scores[name] = probabilities
            results[name] = case_metrics(labels, probabilities, thresholds[name])
            np.savez(
                output / f"{name}_predictions.npz",
                case_ids=np.array(ids),
                labels=labels,
                probabilities=probabilities,
                image_probabilities=image_scores,
            )
            status["completed_models"].append(name)
            save_status()
            print(json.dumps({"model": name, "metrics": results[name]}), flush=True)
        report = {
            "status": "complete",
            "models": results,
            "case_count": len(labels),
            "positive_cases": int(np.count_nonzero(labels)),
            "unit": audit["unit"],
            "paired_intervals": paired_intervals(labels, scores, thresholds),
            "target": audit["target"],
            "no_automatic_deployment": True,
            "interpretation": (
                "External report-code transfer; compare all metrics and uncertainty "
                "before claiming improvement"
            ),
        }
        (output / "result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        status["state"] = "complete"
        save_status()
        return report
    except Exception as error:
        status["state"] = "failed_preserved"
        status["error"] = f"{type(error).__name__}: {error}"
        save_status()
        raise


def main() -> None:
    """Prepare a source-audited cohort separately from one-time model inference."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "evaluate"])
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--cohort", type=Path, default=Path("data/processed/openi_external_v1"))
    parser.add_argument("--output", type=Path, default=Path("reports/metrics/openi_external_v1"))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    root = args.root.resolve()
    cohort = root / args.cohort
    result = (
        prepare(root, cohort)
        if args.stage == "prepare"
        else evaluate(root, cohort, root / args.output, args.device)
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
