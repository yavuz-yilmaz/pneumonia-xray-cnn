"""Inventory NIH pneumonia labels, corrected ages, and prior RSNA patient exposure."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


def inspect_rows(rows: list[dict], mappings: list[dict], exposed: set[str]) -> dict:
    """Validate identifiers and summarize candidate data without assigning splits."""
    images: dict[str, dict] = {}
    patients: set[str] = set()
    adults: list[dict] = []
    exclusions = Counter()
    labels = Counter()
    rsna_names = {record["img_id"] for record in mappings}
    for row in rows:
        name = row["Image Index"]
        if name in images:
            raise ValueError(f"Repeated NIH image identifier: {name}")
        patient = str(int(row["Patient ID"])).zfill(8)
        if name.split("_")[0] != patient:
            raise ValueError("Image filename disagrees with original patient ID")
        age = int(row["Patient Age"])
        findings = set(row["Finding Labels"].split("|"))
        if "No Finding" in findings and len(findings) != 1:
            raise ValueError("Conflicting No Finding and pathology labels")
        images[name] = row
        patients.add(patient)
        labels.update(findings)
        if not 0 <= age <= 120:
            exclusions["implausible_age"] += 1
        elif age < 18:
            exclusions["under_18"] += 1
        else:
            adults.append(
                {
                    "image_id": name,
                    "patient_id": patient,
                    "age_years": age,
                    "report_pneumonia": "Pneumonia" in findings,
                    "no_finding": findings == {"No Finding"},
                    "previous_rsna_exposure": patient in exposed,
                    "available_in_rsna_archive": name in rsna_names,
                }
            )
    if rsna_names - images.keys():
        raise ValueError("NIH metadata does not cover every RSNA mapping")
    positives = [row for row in adults if row["report_pneumonia"]]
    fresh = [row for row in positives if not row["previous_rsna_exposure"]]
    return {
        "images": len(rows),
        "patients": len(patients),
        "report_label_counts": dict(sorted(labels.items())),
        "adult_images": len(adults),
        "adult_patients": len({r["patient_id"] for r in adults}),
        "age_exclusions": dict(exclusions),
        "adult_pneumonia_images": len(positives),
        "adult_pneumonia_patients": len({r["patient_id"] for r in positives}),
        "adult_pneumonia_images_in_rsna": sum(r["available_in_rsna_archive"] for r in positives),
        "adult_pneumonia_images_outside_rsna": sum(
            not r["available_in_rsna_archive"] for r in positives
        ),
        "previously_exposed_rsna_patients": len(exposed),
        "fresh_adult_pneumonia_images": len(fresh),
        "fresh_adult_pneumonia_patients": len({r["patient_id"] for r in fresh}),
        "rsna_adult_images": sum(r["available_in_rsna_archive"] for r in adults),
        "adult_records": adults,
    }


def main() -> None:
    """Save an immutable metadata inventory before choosing a training cohort."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata", type=Path, default=Path("data/raw/nih_metadata/Data_Entry_2017_v2020.csv")
    )
    parser.add_argument(
        "--mapping",
        type=Path,
        default=Path("data/raw/rsna_metadata/pneumonia-challenge-dataset-mappings_2018.json"),
    )
    parser.add_argument(
        "--previous-cohorts",
        type=Path,
        nargs="+",
        default=[
            Path("data/processed/rsna_external_v1/cohort.json"),
            Path("data/processed/rsna_development_v1/cohort.json"),
        ],
    )
    parser.add_argument("--output", type=Path, default=Path("data/processed/nih_inventory_v1"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite an existing inventory")
    with args.metadata.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    mappings = json.loads(args.mapping.read_bytes())
    exposed: set[str] = set()
    for path in args.previous_cohorts:
        exposed.update(row["nih_patient_id"] for row in json.loads(path.read_bytes()))
    result = inspect_rows(rows, mappings, exposed)
    records = result.pop("adult_records")
    paths = [args.metadata, args.mapping, *args.previous_cohorts]
    result["source_sha256"] = {
        path.as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths
    }
    result["limitations"] = [
        "Pneumonia is a report-derived label, not adjudicated clinical ground truth.",
        "Other pathologies and uncertain unlabeled pneumonia may exist among negative labels.",
        "Fresh means absent from recorded RSNA cohorts, not an already locked final test.",
        "Image duplicate auditing and training/test assignment remain pending.",
        "Age is taken from the updated official NIH metadata, not stale RSNA DICOM values.",
    ]
    result["status"] = "Metadata inspected; archive downloads and image audit pending"
    args.output.mkdir(parents=True)
    (args.output / "adult_records.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    result["adult_records_sha256"] = hashlib.sha256(
        (args.output / "adult_records.json").read_bytes()
    ).hexdigest()
    (args.output / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
