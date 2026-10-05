"""Decode and fingerprint completed NIH archives without unsafe tar extraction."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import re
import tarfile
import time
from collections import Counter
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image

from src.data.download_nih import file_sha256, validate_record
from src.data.extract_rsna_cohort import image_signature


def image_name(member: tarfile.TarInfo) -> str:
    """Accept regular NIH images only; never use archive paths as output paths."""
    path = PurePosixPath(member.name)
    if (
        not member.isfile()
        or path.is_absolute()
        or ".." in path.parts
        or len(path.parts) != 2
        or path.parts[0] != "images"
        or not re.fullmatch(r"\d{8}_\d{3}\.png", path.name)
        or not 0 < member.size <= 16 * 1024 * 1024
    ):
        raise ValueError(f"Unexpected NIH archive member: {member.name!r}")
    return path.name


def inspect_image(payload: bytes, row: dict, output: Path) -> tuple[dict, np.ndarray]:
    """Decode original pixels and preserve eligible original PNG bytes unchanged."""
    name = row["Image Index"]
    if not re.fullmatch(r"\d{8}_\d{3}\.png", name):
        raise ValueError("Invalid NIH image identifier")
    patient = str(int(row["Patient ID"])).zfill(8)
    if name.split("_")[0] != patient:
        raise ValueError("NIH image disagrees with metadata patient ID")
    age = int(row["Patient Age"])
    adult = 18 <= age <= 120
    findings = row["Finding Labels"].split("|")
    if "No Finding" in findings and len(findings) != 1:
        raise ValueError("Conflicting report labels")
    with Image.open(io.BytesIO(payload)) as image:
        if image.format != "PNG":
            raise ValueError("NIH source image is not a PNG")
        image.load()
        width, height = image.size
        # The public PNG release is resized to 1024 square. Metadata dimensions
        # describe the original acquisition, not the released PNG.
        if (width, height) != (1024, 1024):
            raise ValueError("Unexpected public NIH PNG dimensions")
        original_width, original_height = int(row["OriginalImage[Width"]), int(row["Height]"])
        if min(original_width, original_height) <= 0:
            raise ValueError("Invalid original acquisition dimensions")
        pixel_hash, phash, thumbnail = image_signature(image)
    digest = hashlib.sha256(payload).hexdigest()
    destination = output / name
    if adult:
        if destination.exists():
            if file_sha256(destination) != digest:
                raise ValueError("Existing output image differs from the source; preserving it")
        else:
            with destination.open("xb") as stream:
                stream.write(payload)
    return {
        "image_id": name,
        "filepath": destination.as_posix() if adult else None,
        "patient_proxy": "NIH:" + patient,
        "nih_patient_id": patient,
        "age_years": age,
        "eligible_adult": adult,
        "label_id": int("Pneumonia" in findings),
        "label": "REPORT_PNEUMONIA" if "Pneumonia" in findings else "NO_REPORTED_PNEUMONIA",
        "source_labels": findings,
        "view_position": row["View Position"],
        "width": width,
        "height": height,
        "metadata_original_width": original_width,
        "metadata_original_height": original_height,
        "sha256": digest,
        "pixel_sha256": pixel_hash,
        "phash": phash,
    }, thumbnail


def extract_archive(
    archive: Path, expected: dict, metadata: dict, output: Path, audit: Path
) -> dict:
    """Verify a downloaded archive and persist a completed per-archive image audit."""
    validate_record(expected)
    if archive.name != expected["name"] or archive.stat().st_size != expected["expected_bytes"]:
        raise ValueError("Archive does not match official listing")
    provenance = json.loads(archive.with_suffix(archive.suffix + ".provenance.json").read_bytes())
    digest = file_sha256(archive)
    if digest != provenance["sha256"] or provenance["source_url"] != expected["url"]:
        raise ValueError("Archive differs from download provenance")
    target = audit / archive.name.removesuffix(".tar.gz")
    completed = target / "image_audit.json"
    if completed.exists():
        previous = json.loads(completed.read_bytes())
        if previous["archive_sha256"] != digest:
            raise ValueError("Completed image audit refers to a different archive")
        for name, checksum in previous["artifact_sha256"].items():
            if file_sha256(target / name) != checksum:
                raise ValueError("Completed image audit artifacts changed")
        return previous
    target.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    records, thumbnails, seen = [], [], set()
    # gzip.GzipFile verifies CRC on reaching EOF. Draining it after tar EOF is
    # essential because tar's end marker may precede the gzip footer.
    with gzip.open(archive, "rb") as compressed:
        with tarfile.open(fileobj=compressed, mode="r|") as source:
            for member in source:
                if member.isdir():
                    continue
                name = image_name(member)
                if name in seen or name not in metadata:
                    raise ValueError("Repeated image or image missing from NIH metadata")
                seen.add(name)
                stream = source.extractfile(member)
                if stream is None:
                    raise ValueError("Cannot read regular image member")
                payload = stream.read(member.size + 1)
                if len(payload) != member.size:
                    raise ValueError("Incomplete source image")
                record, thumbnail = inspect_image(payload, metadata[name], output)
                record["archive"] = archive.name
                records.append(record)
                thumbnails.append(thumbnail)
                if len(records) % 1000 == 0:
                    print(f"{archive.name}: decoded {len(records)} images", flush=True)
        while compressed.read(4 * 1024 * 1024):
            pass
    if not records:
        raise ValueError("Archive contains no NIH images")
    # Only generated audit artifacts are rewritten after an interrupted extraction.
    # Original images and completed audits are preserved and checked above.
    (target / "images.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    np.savez_compressed(target / "thumbnails.npz", thumbnails=np.stack(thumbnails))
    result = {
        "archive": archive.name,
        "archive_sha256": digest,
        "images": len(records),
        "adult_images": sum(row["eligible_adult"] for row in records),
        "adult_label_counts": dict(
            Counter(row["label"] for row in records if row["eligible_adult"])
        ),
        "artifact_sha256": {
            name: file_sha256(target / name) for name in ("images.json", "thumbnails.npz")
        },
        "gzip_crc_verified": True,
        "all_image_pixels_decoded": True,
        "status": "extracted; global patient/duplicate split audit pending",
    }
    completed.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)
    return result


def main() -> None:
    """Process complete downloads, optionally waiting for the ongoing transfer queue."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--archives", type=Path, default=Path("data/raw/nih_archives"))
    parser.add_argument(
        "--metadata", type=Path, default=Path("data/raw/nih_metadata/Data_Entry_2017_v2020.csv")
    )
    parser.add_argument("--output", type=Path, default=Path("data/raw/nih_images"))
    parser.add_argument("--audit", type=Path, default=Path("data/processed/nih_images_v1"))
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_bytes())
    records = manifest["archives"]
    for record in records:
        validate_record(record)
    with args.metadata.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    metadata = {row["Image Index"]: row for row in rows}
    if len(metadata) != len(rows):
        raise ValueError("Duplicate metadata image IDs")
    metadata_sha = file_sha256(args.metadata)
    if metadata_sha != "c69a6daca3549af707ca9cacbdf9f9a7b6a9188e8c61157df653520bb72d8eb1":
        raise ValueError("NIH metadata differs from the prospectively fixed source")
    args.audit.mkdir(parents=True, exist_ok=True)
    contract = args.audit / "source_contract.json"
    expected_contract = {
        "metadata_sha256": metadata_sha,
        "download_manifest_sha256": file_sha256(args.manifest),
    }
    if contract.exists():
        if json.loads(contract.read_bytes()) != expected_contract:
            raise ValueError("Image extraction inputs changed")
    else:
        contract.write_text(json.dumps(expected_contract, indent=2), encoding="utf-8")
    processed = set()
    last_report = 0.0
    while len(processed) < len(records):
        for record in records:
            archive = args.archives / record["name"]
            provenance = archive.with_suffix(archive.suffix + ".provenance.json")
            if record["name"] not in processed and archive.exists() and provenance.exists():
                extract_archive(archive, record, metadata, args.output, args.audit)
                processed.add(record["name"])
        if not args.watch:
            break
        now = time.monotonic()
        if now - last_report >= 60:
            print(f"Extracted {len(processed)}/{len(records)} completed archives", flush=True)
            last_report = now
        if len(processed) < len(records):
            time.sleep(10)
    print(f"Extraction pass completed: {len(processed)}/{len(records)} archives", flush=True)


if __name__ == "__main__":
    main()
