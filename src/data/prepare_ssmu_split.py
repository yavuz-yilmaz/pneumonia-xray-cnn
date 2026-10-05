"""Prepare a conservative filename/duplicate-disjoint SSMU frontal cohort."""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import tarfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

from src.data.clean_split import Components, audit_splits
from src.data.dataset import LABEL_TO_ID
from src.data.download_nih import file_sha256
from src.data.extract_rsna_cohort import image_signature

ARCHIVE_SHA256 = "430127367f26dd7eb4e6a74f19fc2c7649e975a25c3c05df732810e5dc879dba"
CLASS_DIRECTORIES = {"pneumonia": "PNEUMONIA", "norma": "NORMAL"}


def source_record(row: dict) -> dict:
    """Require the exact known filename/layout and retain unverified identity."""
    member = PurePosixPath(row["image_id"])
    match = re.fullmatch(r"(\d+)_(pa|lat)\.png", member.name)
    if (
        member.parts[:2] != ("dataset", "images")
        or len(member.parts) != 4
        or member.parent.name not in CLASS_DIRECTORIES
        or not match
    ):
        raise ValueError("Unknown SSMU source layout")
    label = CLASS_DIRECTORIES[member.parent.name]
    return dict(
        row,
        label=label,
        label_id=LABEL_TO_ID[label],
        patient_proxy=f"ssmu_numeric_{int(match.group(1))}",
        source_view=match.group(2),
        patient_identity_verified=False,
    )


def group_and_select(records: list[dict], pairs: list[dict]) -> tuple:
    """Join stems globally and lateral matches; remove redundant frontal images."""
    components, image_components = Components(len(records)), Components(len(records))
    names = {r["image_id"]: i for i, r in enumerate(records)}
    if len(names) != len(records):
        raise ValueError("Repeated source image IDs")
    stems = {}
    for i, row in enumerate(records):
        proxy = row["patient_proxy"]
        if proxy in stems:
            components.join(i, stems[proxy])
        stems[proxy] = i
    matches = []
    for pair in pairs:
        if pair["reference_source"] != "SSMU":
            raise ValueError("Source overlaps prior cohorts; require a new exclusion protocol")
        first, second = names[pair["ssmu_image_id"]], names[pair["reference_image_id"]]
        if records[first]["label_id"] != records[second]["label_id"]:
            raise ValueError("Conflicting source labels in a duplicate pair")
        components.join(first, second)
        image_components.join(first, second)
        matches.append({"first": first, "second": second, "kind": pair["kind"]})
    frontal = [i for i, r in enumerate(records) if r["source_view"] == "pa"]
    if any(records[i]["mode"] != "L" for i in frontal):
        raise ValueError("Frontal input requires original 8-bit pixels")
    by_duplicate = defaultdict(list)
    for i in frontal:
        by_duplicate[image_components.root(i)].append(i)
    selected = sorted(
        min(indices, key=lambda i: records[i]["image_id"]) for indices in by_duplicate.values()
    )
    return components, matches, selected


def fixed_folds(records: list[dict], components: Components, selected: list[int]) -> dict:
    """Reserve first two fixed seven-fold group partitions without score access."""
    labels = [records[i]["label_id"] for i in selected]
    groups = [components.root(i) for i in selected]
    splitter = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=20261002)
    folds = list(splitter.split(selected, labels, groups))
    test = {selected[i] for i in folds[0][1]}
    val = {selected[i] for i in folds[1][1]}
    splits = {
        "train": sorted(set(selected) - test - val),
        "val": sorted(val),
        "test": sorted(test),
        "excluded": sorted(set(range(len(records))) - set(selected)),
    }
    for name in ("train", "val", "test"):
        if {records[i]["label_id"] for i in splits[name]} != {0, 1}:
            raise ValueError(f"Both classes required in {name}")
    if sorted(i for indices in splits.values() for i in indices) != list(range(len(records))):
        raise ValueError("Incomplete/overlapping fold assignment")
    return splits


def prepare(archive_path: Path, audited: Path, output: Path) -> dict:
    """Verify immutable source evidence and extract only selected frontal PNGs."""
    if output.exists():
        raise FileExistsError("Preserve prior split; inspect its state")
    report = json.loads((audited / "image_audit.json").read_bytes())
    if (
        report["archive_sha256"] != ARCHIVE_SHA256
        or not all(
            report[k]
            for k in (
                "publisher_checksum_verified",
                "gzip_crc_verified",
                "all_image_pixels_decoded",
            )
        )
        or report["cross_source_matched_images"] != 0
    ):
        raise ValueError("Source integrity/overlap audit is not admissible")
    input_paths = [
        archive_path,
        audited / "image_audit.json",
        Path(__file__),
        Path("docs/ssmu_development_protocol.md"),
    ]
    for name, digest in report["artifact_sha256"].items():
        path = audited / name
        if file_sha256(path) != digest:
            raise ValueError("Audited source artifact changed")
        input_paths.append(path)
    inputs = {str(p): file_sha256(p) for p in input_paths}
    if inputs[str(archive_path)] != ARCHIVE_SHA256:
        raise ValueError("Publisher archive changed")
    records = [source_record(r) for r in json.loads((audited / "images.json").read_bytes())]
    pairs = json.loads((audited / "duplicate_matches.json").read_bytes())
    components, matches, selected = group_and_select(records, pairs)
    splits = fixed_folds(records, components, selected)
    audit = audit_splits(records, splits, components, matches)
    output.mkdir(parents=True)
    (output / "registration.json").write_text(
        json.dumps(
            {
                "input_sha256": inputs,
                "protocol": "docs/ssmu_development_protocol.md",
                "model_scores_observed": False,
                "patient_identity_verified": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    needed = {records[i]["image_id"]: i for i in selected}
    with tarfile.open(archive_path, "r|gz") as archive:
        for entry in archive:
            if entry.name not in needed:
                continue
            i = needed.pop(entry.name)
            stream = archive.extractfile(entry)
            if stream is None or not entry.isfile() or entry.size > 100_000_000:
                raise ValueError("Unexpected selected TAR entry")
            with stream:
                payload = stream.read()
            target = (
                output
                / "images"
                / PurePosixPath(entry.name).parent.name
                / PurePosixPath(entry.name).name
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            target.open("xb").write(payload)
            if file_sha256(target) != records[i]["sha256"]:
                raise ValueError("Extracted image differs")
            with Image.open(io.BytesIO(payload)) as image:
                image.load()
                pixels, phash, _ = image_signature(image)
                if (
                    image.mode != "L"
                    or pixels != records[i]["pixel_sha256"]
                    or phash != records[i]["phash"]
                ):
                    raise ValueError("Extracted source pixels differ")
            records[i]["filepath"] = target.as_posix()
    if needed:
        raise ValueError("Selected images missing from source archive")
    manifest_hashes = {}
    fields = [
        "filepath",
        "split",
        "label",
        "label_id",
        "patient_proxy",
        "group_id",
        "sha256",
        "pixel_sha256",
        "source_member",
        "patient_identity_verified",
    ]
    for split, indices in splits.items():
        path = output / f"{split}_manifest.csv"
        with path.open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for i in indices:
                row = records[i]
                writer.writerow(
                    {
                        "filepath": row.get("filepath", ""),
                        "split": split,
                        "label": row["label"],
                        "label_id": row["label_id"],
                        "patient_proxy": row["patient_proxy"],
                        "group_id": components.root(i),
                        "sha256": row["sha256"],
                        "pixel_sha256": row["pixel_sha256"],
                        "source_member": row["image_id"],
                        "patient_identity_verified": False,
                    }
                )
        manifest_hashes[split] = file_sha256(path)
    audit.update(
        task="ssmu_source_pneumonia",
        class_to_idx=LABEL_TO_ID,
        patient_identity_verified=False,
        clinical_labels_verified=False,
        manifest_sha256=manifest_hashes,
        input_sha256=inputs,
        source_images=len(records),
        eligible_frontal_before_deduplication=sum(r["source_view"] == "pa" for r in records),
        frontal_duplicates_removed=sum(r["source_view"] == "pa" for r in records) - len(selected),
        identity_rule="Global numeric stems across classes/views, plus all recorded duplicates",
        limitations=[
            "Patient IDs and clinical labels unverified",
            "Conservative duplicate screen cannot prove complete independence",
        ],
        selected_label_counts=dict(Counter(records[i]["label"] for i in selected)),
    )
    (output / "split_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive", type=Path, default=Path("data/raw/pneumonia_norm_2021/dataset.tar.gz")
    )
    parser.add_argument("--audited", type=Path, default=Path("data/processed/ssmu_images_v1"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/ssmu_frontal_v1"))
    args = parser.parse_args()
    print(json.dumps(prepare(args.archive, args.audited, args.output), indent=2), flush=True)


if __name__ == "__main__":
    main()
