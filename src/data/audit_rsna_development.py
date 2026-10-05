"""Audit RSNA development against reserved external images before pretraining."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image

from src.data.clean_split import connect_records
from src.data.extract_rsna_cohort import image_signature


def read_verified(directory: Path) -> list[dict]:
    """Verify the image manifest, reserved identities and source bytes."""
    protocol = json.loads((directory / "protocol.json").read_bytes())
    cohort_bytes = (directory / "cohort.json").read_bytes()
    if hashlib.sha256(cohort_bytes).hexdigest() != protocol["cohort_sha256"]:
        raise ValueError("Cohort changed")
    audit = json.loads((directory / "image_audit.json").read_bytes())
    content = (directory / "images.json").read_bytes()
    if (
        audit["cohort_sha256"] != protocol["cohort_sha256"]
        or hashlib.sha256(content).hexdigest() != audit["images_manifest_sha256"]
    ):
        raise ValueError("Image audit mismatch")
    if audit["overlaps"]:
        raise ValueError("Image extraction found overlaps requiring review")
    originals = {r["nih_patient_id"]: r for r in json.loads(cohort_bytes)}
    records = json.loads(content)
    if len(records) != len(originals):
        raise ValueError("Patient count mismatch")
    for row in records:
        original = originals[row["nih_patient_id"]]
        if any(row.get(key) != value for key, value in original.items()):
            raise ValueError("Source identity or label changed")
        path = Path(row["filepath"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError("Image changed")
        with Image.open(path) as image:
            pixel_hash, phash, thumbnail = image_signature(image)
        if pixel_hash != row["pixel_sha256"]:
            raise ValueError("Decoded image changed")
        row.update(patient_proxy="NIH:" + row["nih_patient_id"], phash=phash, thumbnail=thumbnail)
    return records


def main() -> None:
    """Fail closed on duplicates spanning train/validation/external patient groups."""
    directory = Path("data/processed/rsna_development_v1")
    target = directory / "development_audit.json"
    if target.exists():
        raise FileExistsError("Development audit already exists")
    development = read_verified(directory)
    external = read_verified(Path("data/processed/rsna_external_v1"))
    for row in external:
        row["split"] = "external"
    records = development + external
    components, matches = connect_records(records)
    members = {}
    for index, row in enumerate(records):
        members.setdefault(components.root(index), []).append(index)
    conflicts = []
    for indices in members.values():
        splits = {records[i]["split"] for i in indices}
        if len(splits) > 1:
            conflicts.append(
                {"splits": sorted(splits), "images": [records[i]["nih_image_id"] for i in indices]}
            )
    report = {
        "development_manifest_sha256": hashlib.sha256(
            (directory / "images.json").read_bytes()
        ).hexdigest(),
        "external_manifest_sha256": hashlib.sha256(
            Path("data/processed/rsna_external_v1/images.json").read_bytes()
        ).hexdigest(),
        "development_count": len(development),
        "external_count": len(external),
        "cross_split_components": conflicts,
        "duplicate_pairs": matches,
        "status": "passed" if not conflicts else "requires_review",
        "screening": "NIH patient IDs, byte hashes, decoded hashes, pHash<=6 and thumbnail correlation>=0.995",
    }
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if conflicts:
        raise ValueError("Do not pretrain: development overlaps reserved or validation data")


if __name__ == "__main__":
    main()
