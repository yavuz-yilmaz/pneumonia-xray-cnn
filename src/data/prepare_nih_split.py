"""Lock adult NIH splits using original patients, duplicates and historical exposure."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from src.data.clean_split import Components, audit_splits, connect_records
from src.data.download_nih import file_sha256
from src.data.extract_rsna_cohort import image_signature


def assign_nih_splits(
    records: list[dict], components: Components, historical_roles: dict[str, str], seed: int
) -> tuple[dict[str, list[int]], dict[int, str]]:
    """Assign connected groups conservatively; never turn observed patients into fresh test."""
    if not set(historical_roles.values()) <= {"train", "val", "external"}:
        raise ValueError("Unknown historical patient role")
    members = defaultdict(list)
    pixels = defaultdict(set)
    for index, row in enumerate(records):
        members[components.root(index)].append(index)
        if row.get("source_family") == "nih":
            pixels[row["pixel_sha256"]].add(row["label_id"])
    ambiguous = {pixel for pixel, labels in pixels.items() if len(labels) > 1}
    splits = {name: [] for name in ("train", "val", "test", "excluded")}
    reasons: dict[int, str] = {}
    fresh = {0: [], 1: []}
    for indices in members.values():
        adults = [i for i in indices if records[i].get("eligible_adult", False)]
        prior = {
            historical_roles[records[i]["nih_patient_id"]]
            for i in indices
            if records[i].get("nih_patient_id") in historical_roles
        }
        reason = None
        if any(records[i].get("source_family") == "legacy_pediatric" for i in indices):
            reason = "touches_observed_pediatric_data"
        elif "external" in prior:
            reason = "touches_reserved_rsna_external_patient"
        elif len(prior) > 1:
            reason = "conflicting_historical_roles"
        elif any(records[i]["pixel_sha256"] in ambiguous for i in indices):
            reason = "identical_pixels_with_conflicting_pneumonia_labels"
        elif not adults:
            reason = "no_eligible_adult_acquisition"
        if reason:
            splits["excluded"].extend(indices)
            reasons.update(dict.fromkeys(indices, reason))
            continue
        for i in indices:
            if i not in adults:
                splits["excluded"].append(i)
                reasons[i] = "age_outside_adult_cohort"
        if prior:
            splits[next(iter(prior))].extend(adults)
            continue
        patient_key = ":".join(sorted({records[i]["nih_patient_id"] for i in indices}))
        rank = hashlib.sha256(f"{seed}:{patient_key}".encode()).hexdigest()
        positive = int(any(records[i]["label_id"] == 1 for i in adults))
        fresh[positive].append((rank, adults))
    for groups in fresh.values():
        ordered = sorted(groups)
        test_end = round(len(ordered) * 0.15)
        val_end = test_end + round(len(ordered) * 0.15)
        for position, (_, indices) in enumerate(ordered):
            split = "test" if position < test_end else "val" if position < val_end else "train"
            splits[split].extend(indices)
    for indices in splits.values():
        indices.sort()
    flat = [i for indices in splits.values() for i in indices]
    if sorted(flat) != list(range(len(records))):
        raise ValueError("NIH split omitted or repeated source records")
    for split in ("train", "val", "test"):
        if not splits[split] or {records[i]["label_id"] for i in splits[split]} != {0, 1}:
            raise ValueError(f"Both labels required in {split}")
    return splits, reasons


def load_extracted(directory: Path, manifest: Path, metadata_path: Path) -> list[dict]:
    """Require every archive audit and exact official metadata coverage."""
    contract = json.loads((directory / "source_contract.json").read_bytes())
    if (
        file_sha256(metadata_path) != contract["metadata_sha256"]
        or file_sha256(manifest) != contract["download_manifest_sha256"]
    ):
        raise ValueError("Extraction source contract changed")
    expected = json.loads(manifest.read_bytes())["archives"]
    with metadata_path.open(newline="", encoding="utf-8") as stream:
        metadata = {row["Image Index"]: row for row in csv.DictReader(stream)}
    records, names = [], set()
    for archive in expected:
        source = directory / archive["name"].removesuffix(".tar.gz")
        report = json.loads((source / "image_audit.json").read_bytes())
        if not report["gzip_crc_verified"] or not report["all_image_pixels_decoded"]:
            raise ValueError("Source archive has not passed image integrity checks")
        for filename, digest in report["artifact_sha256"].items():
            if file_sha256(source / filename) != digest:
                raise ValueError("Extraction artifact changed")
        rows = json.loads((source / "images.json").read_bytes())
        with np.load(source / "thumbnails.npz", allow_pickle=False) as compressed:
            thumbnails = compressed["thumbnails"]
        if len(rows) != report["images"] or thumbnails.shape != (len(rows), 64, 64):
            raise ValueError("Thumbnail or image inventory length mismatch")
        for row, thumbnail in zip(rows, thumbnails, strict=True):
            name = row["image_id"]
            if name in names or name not in metadata:
                raise ValueError("Repeated or unknown NIH image across source archives")
            original = metadata[name]
            patient = str(int(original["Patient ID"])).zfill(8)
            age = int(original["Patient Age"])
            label = int("Pneumonia" in original["Finding Labels"].split("|"))
            if (
                row["nih_patient_id"] != patient
                or row["patient_proxy"] != "NIH:" + patient
                or row["age_years"] != age
                or row["eligible_adult"] != (18 <= age <= 120)
                or row["label_id"] != label
            ):
                raise ValueError("Extracted record disagrees with official NIH metadata")
            names.add(name)
            records.append(dict(row, source_family="nih", thumbnail=thumbnail))
    if names != metadata.keys():
        raise ValueError("Downloaded image inventory does not cover official NIH metadata")
    return sorted(records, key=lambda row: row["image_id"])


def main() -> None:
    """Persist prospective manifests only after global image/group auditing."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--extraction", type=Path, default=Path("data/processed/nih_images_v1"))
    parser.add_argument(
        "--metadata", type=Path, default=Path("data/raw/nih_metadata/Data_Entry_2017_v2020.csv")
    )
    parser.add_argument("--output", type=Path, default=Path("data/processed/nih_pneumonia_v1"))
    parser.add_argument("--seed", type=int, default=20261001)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("NIH split already exists; refusing to overwrite")
    records = load_extracted(args.extraction, args.manifest, args.metadata)
    historical = {}
    cohort_paths = [
        Path("data/processed/rsna_external_v1/cohort.json"),
        Path("data/processed/rsna_development_v1/cohort.json"),
    ]
    for path in cohort_paths:
        protocol = json.loads((path.parent / "protocol.json").read_bytes())
        if file_sha256(path) != protocol["cohort_sha256"]:
            raise ValueError("Historical cohort changed since registration")
        for row in json.loads(path.read_bytes()):
            patient = row["nih_patient_id"]
            role = "external" if "external" in path.parent.name else row["split"]
            if patient in historical and historical[patient] != role:
                raise ValueError("Historical patient roles conflict")
            historical[patient] = role
    pediatric = Path("data/processed/clean_v1/source_inventory.json")
    for row in json.loads(pediatric.read_bytes()):
        if file_sha256(Path(row["filepath"])) != row["sha256"]:
            raise ValueError("Previously observed pediatric source changed")
        with Image.open(row["filepath"]) as image:
            pixels, phash, thumbnail = image_signature(image)
        if pixels != row["pixel_sha256"]:
            raise ValueError("Previously observed pediatric image pixels changed")
        records.append(
            dict(row, source_family="legacy_pediatric", phash=phash, thumbnail=thumbnail)
        )
    print(f"Auditing {len(records)} source fingerprints and original patient groups", flush=True)
    components, matches = connect_records(records)
    splits, reasons = assign_nih_splits(records, components, historical, args.seed)
    report = audit_splits(records, splits, components, matches)
    report.update(
        {
            "seed": args.seed,
            "target": "NIH report-derived Pneumonia label",
            "class_to_idx": {"NO_REPORTED_PNEUMONIA": 0, "REPORT_PNEUMONIA": 1},
            "age_source": "Updated official NIH metadata; 18–120 years",
            "historical_patient_roles_preserved": True,
            "fresh_test_exposed_patient_overlap": len(
                {records[i]["nih_patient_id"] for i in splits["test"]} & historical.keys()
            ),
            "source_sha256": {
                path.as_posix(): file_sha256(path)
                for path in [args.metadata, args.manifest, pediatric, *cohort_paths]
            },
            "near_duplicate_rule": "phash_hamming<=6 and thumbnail_correlation>=0.995",
            "limitations": [
                "Report-derived labels are not adjudicated clinical pneumonia ground truth.",
                "Near-duplicate screening is conservative and not exhaustive identity proof.",
                "This is a single-institution adult dataset, not independent clinical validation.",
            ],
        }
    )
    if report["fresh_test_exposed_patient_overlap"]:
        raise ValueError("Fresh test contains previously exposed patients")
    args.output.mkdir(parents=True)
    columns = [
        "filepath",
        "split",
        "label",
        "label_id",
        "patient_proxy",
        "group_id",
        "sha256",
        "pixel_sha256",
        "image_id",
        "age_years",
        "exclusion_reason",
    ]
    report["manifest_sha256"] = {}
    for split, indices in splits.items():
        path = args.output / f"{split}_manifest.csv"
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            for i in indices:
                row = records[i]
                entry = {key: row.get(key, "") for key in columns}
                entry.update(
                    split=split,
                    group_id=components.root(i),
                    exclusion_reason=reasons.get(i, ""),
                )
                writer.writerow(entry)
        report["manifest_sha256"][split] = file_sha256(path)
    (args.output / "split_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    serializable = [{k: v for k, v in row.items() if k != "thumbnail"} for row in records]
    (args.output / "source_inventory.json").write_text(
        json.dumps(serializable, indent=2), encoding="utf-8"
    )
    (args.output / "duplicate_pairs.json").write_text(
        json.dumps(matches, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
