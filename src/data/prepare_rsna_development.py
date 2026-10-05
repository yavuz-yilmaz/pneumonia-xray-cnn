"""Reserve a separate RSNA development cohort for opacity feature pretraining."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from src.data.prepare_rsna_cohort import cohort_records


def main() -> None:
    """Exclude every external patient's acquisitions before selecting development data."""
    metadata = Path("data/raw/rsna_metadata")
    external = Path("data/processed/rsna_external_v1")
    output = Path("data/processed/rsna_development_v1")
    if output.exists():
        raise FileExistsError("Development cohort already locked")
    external_bytes = (external / "cohort.json").read_bytes()
    external_protocol = json.loads((external / "protocol.json").read_bytes())
    if hashlib.sha256(external_bytes).hexdigest() != external_protocol["cohort_sha256"]:
        raise ValueError("Reserved external cohort changed")
    excluded = {r["nih_patient_id"] for r in json.loads(external_bytes)}
    paths = [
        metadata / name
        for name in (
            "pneumonia-challenge-dataset-mappings_2018.json",
            "pneumonia-challenge-annotations-adjudicated-kaggle_2018.json",
        )
    ]
    records = cohort_records(
        json.loads(paths[0].read_bytes()),
        json.loads(paths[1].read_bytes()),
        per_class=1007,
        seed=20260916,
        excluded_patients=excluded,
        normal_multiplier=2,
    )
    for label in (0, 1):
        rows = sorted(
            (r for r in records if r["label_id"] == label),
            key=lambda r: hashlib.sha256(("split-v1:" + r["nih_patient_id"]).encode()).hexdigest(),
        )
        for index, row in enumerate(rows):
            row["split"] = "val" if index < round(len(rows) * 0.2) else "train"
    if {r["nih_patient_id"] for r in records} & excluded:
        raise ValueError("Development includes reserved external patient")
    output.mkdir(parents=True)
    cohort = output / "cohort.json"
    cohort.write_text(json.dumps(records, indent=2), encoding="utf-8")
    protocol = {
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "cohort_sha256": hashlib.sha256(cohort.read_bytes()).hexdigest(),
        "reserved_external_cohort_sha256": external_protocol["cohort_sha256"],
        "metadata_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "purpose": "Development-only adult opacity pretraining, then pediatric fine-tuning; distinct labels",
        "unique_patients": len(records),
        "reserved_patient_overlap": 0,
        "counts": {
            s: {
                str(c): sum(r["split"] == s and r["label_id"] == c for r in records) for c in (0, 1)
            }
            for s in ("train", "val")
        },
        "selection": "One hash-ranked acquisition per patient; all 1007 eligible opacity patients and 2014 hash-ranked normal patients; class-stratified 80/20 patient split",
        "image_audit_status": "Pending decoding and duplicate audit before training",
        "limitations": [
            "Adult opacity is not clinical pneumonia",
            "Excludes other abnormal images",
            "Prior pediatric and external benchmark scores have been observed",
        ],
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(json.dumps(protocol, indent=2))


if __name__ == "__main__":
    main()
