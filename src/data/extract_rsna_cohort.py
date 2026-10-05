"""Extract locked RSNA DICOM images without changing their decoded 8-bit pixels."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps


def image_signature(image: Image.Image) -> tuple[str, int, np.ndarray]:
    """Match the pediatric audit's decoded grayscale and perceptual fingerprints."""
    gray = ImageOps.exif_transpose(image).convert("L")
    pixels = np.asarray(gray)
    pixel_hash = hashlib.sha256(str(pixels.shape).encode() + pixels.tobytes()).hexdigest()
    small = np.asarray(gray.resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float32)
    coefficients = cv2.dct(small)[:8, :8].flatten()
    bits = coefficients > np.median(coefficients[1:])
    bits[0] = False
    phash = int.from_bytes(np.packbits(bits).tobytes(), "big")
    thumbnail = np.asarray(gray.resize((64, 64), Image.Resampling.BILINEAR))
    return pixel_hash, phash, thumbnail


def main() -> None:
    """Verify cohort/archive hashes, decode selected entries, and screen overlaps."""
    import pydicom

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, default=Path("data/processed/rsna_external_v1"))
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path(
            "data/raw/rsna_metadata/pneumonia-challenge-dataset-adjudicated-kaggle_2018.zip"
        ),
    )
    parser.add_argument("--images", type=Path, default=Path("data/raw/rsna_external_v1"))
    parser.add_argument(
        "--inventory", type=Path, default=Path("data/processed/clean_v1/source_inventory.json")
    )
    args = parser.parse_args()
    if (args.cohort / "image_audit.json").exists():
        raise FileExistsError("External image audit already exists")
    protocol = json.loads((args.cohort / "protocol.json").read_bytes())
    content = (args.cohort / "cohort.json").read_bytes()
    if hashlib.sha256(content).hexdigest() != protocol["cohort_sha256"]:
        raise ValueError("Locked cohort changed")
    provenance = json.loads(args.archive.with_suffix(".provenance.json").read_bytes())
    with args.archive.open("rb") as stream:
        # Chunked hashing also supports the project's Python 3.10 minimum.
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != provenance["sha256"]:
        raise ValueError("Archive changed since download")
    originals = json.loads(args.inventory.read_bytes())
    pediatric_hashes = {r["pixel_sha256"] for r in originals}
    records = json.loads(content)
    args.images.mkdir(parents=True, exist_ok=True)
    seen, matches, converted = {}, [], []
    with zipfile.ZipFile(args.archive) as archive:
        entries = {}
        for entry in archive.infolist():
            name = Path(entry.filename).stem
            if name in entries:
                raise ValueError("Duplicate SOP filename in archive")
            entries[name] = entry
        for index, record in enumerate(records):
            entry = entries[record["sop_instance_uid"]]
            payload = archive.read(entry)  # ZIP CRC verified; no archive path is extracted.
            dicom = pydicom.dcmread(io.BytesIO(payload))
            if str(dicom.SOPInstanceUID) != record["sop_instance_uid"]:
                raise ValueError("DICOM does not match selected SOP")
            pixels = dicom.pixel_array
            if (
                pixels.dtype != np.uint8
                or pixels.ndim != 2
                or dicom.PhotometricInterpretation != "MONOCHROME2"
            ):
                raise ValueError(
                    "Unexpected DICOM pixel encoding; explicit conversion review required"
                )
            if any(
                key in dicom
                for key in ("RescaleSlope", "RescaleIntercept", "WindowCenter", "WindowWidth")
            ):
                raise ValueError("Unexpected DICOM display transform; review before decoding")
            image = Image.fromarray(pixels)
            pixel_hash, phash, thumbnail = image_signature(image)
            if pixel_hash in seen or pixel_hash in pediatric_hashes:
                matches.append({"external": record["nih_image_id"], "kind": "exact_pixel_overlap"})
            seen[pixel_hash] = record["nih_image_id"]
            for original in originals:
                if (phash ^ original["phash"]).bit_count() > 6:
                    continue
                with Image.open(original["filepath"]) as source:
                    _, _, other = image_signature(source)
                correlation = float(np.corrcoef(thumbnail.flatten(), other.flatten())[0, 1])
                if correlation >= 0.995:
                    matches.append(
                        {
                            "external": record["nih_image_id"],
                            "pediatric": original["filepath"],
                            "kind": "near_duplicate",
                            "correlation": correlation,
                        }
                    )
            # Select our own filename; archive directory names never reach the filesystem.
            destination = args.images / (record["rsna_image_id"] + ".png")
            if (
                destination.name != record["rsna_image_id"] + ".png"
                or destination.parent.resolve() != args.images.resolve()
            ):
                raise ValueError("Unsafe image identifier")
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            png = buffer.getvalue()
            if destination.exists() and destination.read_bytes() != png:
                raise ValueError("Existing extracted image differs")
            if not destination.exists():
                destination.write_bytes(png)
            converted.append(
                dict(
                    record,
                    filepath=destination.as_posix(),
                    sha256=hashlib.sha256(png).hexdigest(),
                    pixel_sha256=pixel_hash,
                    source_dicom_sha256=hashlib.sha256(payload).hexdigest(),
                )
            )
            if (index + 1) % 100 == 0:
                print(f"Decoded and screened {index + 1}/{len(records)}", flush=True)
    audit = {
        "cohort_sha256": protocol["cohort_sha256"],
        "archive_sha256": provenance["sha256"],
        "pediatric_inventory_sha256": hashlib.sha256(args.inventory.read_bytes()).hexdigest(),
        "images": len(converted),
        "unique_patients": len({r["nih_patient_id"] for r in converted}),
        "overlaps": matches,
        "decoding": "Original uint8 MONOCHROME2 pixels, lossless PNG; no window or rescale transform",
        "limitations": "Exact/near-duplicate screening does not prove absence of every possible shared acquisition.",
    }
    manifest = args.cohort / "images.json"
    manifest.write_text(json.dumps(converted, indent=2), encoding="utf-8")
    audit["images_manifest_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    (args.cohort / "image_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    if matches:
        raise ValueError("External overlap detected; do not evaluate until reviewed")


if __name__ == "__main__":
    main()
