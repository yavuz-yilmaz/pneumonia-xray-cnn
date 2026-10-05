"""Lock a patient-distinct RSNA external cohort from official annotation metadata.

This is an adult lung-opacity transfer benchmark, not a new pediatric pneumonia
ground truth. No model predictions are read by this module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


def cohort_records(
    mappings: list[dict],
    annotations: dict,
    per_class: int,
    seed: int,
    excluded_patients: set[str] | None = None,
    normal_multiplier: int = 1,
) -> list[dict]:
    """Choose one acquisition per NIH patient before balancing the two classes."""
    by_sop = {record["SOPInstanceUID"]: record for record in mappings}
    if len(by_sop) != len(mappings):
        raise ValueError("Official mapping has duplicate SOP identifiers")
    calculated = [group for group in annotations["labelGroups"] if group["name"] == "Calculated"]
    if len(calculated) != 1:
        raise ValueError("Expected one adjudicated Calculated label group")
    names = {label["id"]: label["name"].strip() for label in calculated[0]["labels"]}
    labels = defaultdict(set)
    for dataset in annotations["datasets"]:
        for row in dataset["annotations"]:
            if row["labelId"] in names:
                labels[row["SOPInstanceUID"]].add(names[row["labelId"]])
    patients = defaultdict(list)
    for sop, values in labels.items():
        if len(values) != 1:
            raise ValueError(f"Conflicting final labels for {sop}")
        source = by_sop[sop]
        patient_id = source["img_id"].split("_")[0]
        if excluded_patients and patient_id in excluded_patients:
            continue
        patients[patient_id].append(
            {
                "nih_patient_id": patient_id,
                "nih_image_id": source["img_id"],
                "rsna_image_id": source["subset_img_id"],
                "sop_instance_uid": sop,
                "source_label": next(iter(values)),
            }
        )

    def rank(record: dict) -> str:
        return hashlib.sha256(f"{seed}:{record['nih_image_id']}".encode()).hexdigest()

    # Choosing a patient's acquisition before class filtering avoids preferentially
    # picking its positive image. Other-abnormal cases are outside this two-class task.
    representatives = [min(records, key=rank) for records in patients.values()]
    selected = []
    for label_id, label in enumerate(("Normal", "Lung Opacity")):
        eligible = sorted((r for r in representatives if r["source_label"] == label), key=rank)
        count = per_class * normal_multiplier if label_id == 0 else per_class
        if len(eligible) < count:
            raise ValueError(f"Only {len(eligible)} eligible patients for {label}")
        selected.extend(dict(record, label_id=label_id) for record in eligible[:count])
    if len({r["nih_patient_id"] for r in selected}) != len(selected):
        raise ValueError("External cohort contains repeated patients")
    return selected


def main() -> None:
    """Persist an immutable cohort before accessing predictions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=Path("data/raw/rsna_metadata"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/rsna_external_v1"))
    parser.add_argument("--per-class", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260916)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("External cohort already locked; refusing overwrite")
    if args.per_class < 1:
        parser.error("--per-class must be positive")
    mapping_path = args.metadata / "pneumonia-challenge-dataset-mappings_2018.json"
    annotation_path = args.metadata / "pneumonia-challenge-annotations-adjudicated-kaggle_2018.json"
    records = cohort_records(
        json.loads(mapping_path.read_bytes()),
        json.loads(annotation_path.read_bytes()),
        args.per_class,
        args.seed,
    )
    args.output.mkdir(parents=True)
    cohort = args.output / "cohort.json"
    cohort.write_text(json.dumps(records, indent=2), encoding="utf-8")
    protocol = {
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "per_class": args.per_class,
        "unique_patients": len(records),
        "cohort_sha256": hashlib.sha256(cohort.read_bytes()).hexdigest(),
        "metadata_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (mapping_path, annotation_path)
        },
        "source": "https://www.rsna.org/rsnai/ai-image-challenge/rsna-pneumonia-detection-challenge-2018",
        "selection": "Hash-ranked single acquisition per NIH patient, then 500 per class by default; no model scores",
        "purpose": "Locked external evaluation only; reserve these NIH patients from all future training and calibration",
        "limitations": [
            "Adult radiographic lung opacity is not equivalent to pediatric clinical pneumonia.",
            "Balanced case-control sampling does not estimate real-population precision or accuracy.",
            "Other abnormal images are excluded; this is not a clinical triage cohort.",
            "TorchXRayVision all-weights models include NIH/RSNA pretraining and must not be claimed independent on this cohort.",
        ],
        "status": "Metadata locked; image extraction and overlap audit pending",
    }
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(json.dumps(protocol, indent=2))


if __name__ == "__main__":
    main()
