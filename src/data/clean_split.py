"""Conservative group-disjoint splits with exact and near-duplicate auditing.

Filename identifiers are proxies, not verified patient identifiers. In particular,
all ``personN`` files are grouped together even across bacteria/virus names. This
may over-group different patients, but avoids assuming those namespaces are distinct.
The original test set stays untouched; development groups touching it are excluded.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps
from sklearn.model_selection import StratifiedGroupKFold

from src.data.dataset import LABEL_TO_ID, SUPPORTED_IMAGE_EXTENSIONS


def group_id(path: Path) -> str:
    """Extract a conservative source identifier; fail closed for unknown names."""
    person = re.match(r"^(person\d+)_", path.stem)
    normal = re.match(r"^((?:NORMAL2-)?IM-\d+)-", path.stem)
    match = person or normal
    if match is None:
        raise ValueError(f"Unknown patient filename convention: {path.name}")
    return match.group(1)


def fingerprint(path: Path) -> dict:
    """Fingerprint bytes, decoded pixels, and perceptual image structure."""
    content = path.read_bytes()
    with Image.open(path) as source:
        gray = ImageOps.exif_transpose(source).convert("L")
        pixels = np.asarray(gray)
        thumbnail = np.asarray(gray.resize((64, 64), Image.Resampling.BILINEAR))
        small = np.asarray(gray.resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float32)
    coefficients = cv2.dct(small)[:8, :8].flatten()
    bits = coefficients > np.median(coefficients[1:])
    bits[0] = False
    perceptual = int.from_bytes(np.packbits(bits).tobytes(), "big")
    pixel_hash = hashlib.sha256(str(pixels.shape).encode() + pixels.tobytes()).hexdigest()
    return {
        "filepath": path.as_posix(),
        "source_split": path.parent.parent.name,
        "label": path.parent.name,
        "label_id": LABEL_TO_ID[path.parent.name],
        "patient_proxy": group_id(path),
        "sha256": hashlib.sha256(content).hexdigest(),
        "pixel_sha256": pixel_hash,
        "phash": perceptual,
        "thumbnail": thumbnail,
    }


class Components:
    """Union-find over patient proxies and image duplicates."""

    def __init__(self, count: int) -> None:
        self.parents = list(range(count))

    def root(self, index: int) -> int:
        while self.parents[index] != index:
            self.parents[index] = self.parents[self.parents[index]]
            index = self.parents[index]
        return index

    def join(self, first: int, second: int) -> None:
        left, right = self.root(first), self.root(second)
        self.parents[max(left, right)] = min(left, right)


def connect_records(records: list[dict]) -> tuple[Components, list[dict]]:
    """Connect exact matches and conservative perceptual near-duplicate pairs.

    Near duplicates require pHash Hamming distance <= 6 AND normalized pixel
    correlation >= .995 on 64px thumbnails. This is a documented screening rule,
    not a proof that every visually similar acquisition has been detected.
    """
    components = Components(len(records))
    matches: list[dict] = []
    for key in ("patient_proxy", "sha256", "pixel_sha256"):
        seen: dict[str, int] = {}
        for index, record in enumerate(records):
            value = record[key]
            if value in seen:
                other = seen[value]
                components.join(index, other)
                if key != "patient_proxy":
                    matches.append({"first": other, "second": index, "kind": key})
            else:
                seen[value] = index

    # Seven disjoint bands guarantee a shared band for a <=6-bit match.
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        phash = record["phash"]
        candidates: set[int] = set()
        bands = [(band, (phash >> (band * 9)) & (1023 if band == 6 else 511)) for band in range(7)]
        for band in bands:
            candidates.update(buckets[band])
        image = record["thumbnail"].astype(np.float32).flatten()
        image -= image.mean()
        norm = float(np.linalg.norm(image))
        for other in sorted(candidates):
            previous = records[other]
            if record["pixel_sha256"] == previous["pixel_sha256"]:
                continue
            distance = (phash ^ previous["phash"]).bit_count()
            if distance > 6:
                continue
            other_image = previous["thumbnail"].astype(np.float32).flatten()
            other_image -= other_image.mean()
            denominator = norm * float(np.linalg.norm(other_image))
            correlation = float(np.dot(image, other_image) / denominator) if denominator else 0.0
            if correlation >= 0.995:
                components.join(index, other)
                matches.append(
                    {
                        "first": other,
                        "second": index,
                        "kind": "near_duplicate",
                        "phash_distance": distance,
                        "correlation": correlation,
                    }
                )
        for band in bands:
            buckets[band].append(index)
    return components, matches


def assign_splits(records: list[dict], components: Components, seed: int) -> dict[str, list[int]]:
    """Reserve official test, remove connected development groups, group-split rest."""
    test = [i for i, r in enumerate(records) if r["source_split"] == "test"]
    test_groups = {components.root(i) for i in test}
    development: list[int] = []
    excluded: list[int] = []
    seen_pixels: set[str] = set()
    for index, record in enumerate(records):
        if record["source_split"] == "test":
            continue
        if components.root(index) in test_groups or record["pixel_sha256"] in seen_pixels:
            excluded.append(index)
            continue
        seen_pixels.add(record["pixel_sha256"])
        development.append(index)
    labels = [records[i]["label_id"] for i in development]
    groups = [components.root(i) for i in development]
    # Fixed first fold, chosen before any training/metric inspection (~20% validation).
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    train_positions, val_positions = next(splitter.split(development, labels, groups))
    return {
        "train": [development[i] for i in train_positions],
        "val": [development[i] for i in val_positions],
        "test": test,
        "excluded": excluded,
    }


def audit_splits(
    records: list[dict], splits: dict[str, list[int]], components: Components, matches: list[dict]
) -> dict:
    """Assert all documented identity/duplicate groups are split-disjoint."""
    result: dict = {"overlaps": {}, "counts": {}}
    names = ("train", "val", "test")
    for split, indices in splits.items():
        result["counts"][split] = {
            "images": len(indices),
            "labels": dict(Counter(records[i]["label"] for i in indices)),
            "groups": len({components.root(i) for i in indices}),
        }
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        values = {}
        for key in ("patient_proxy", "sha256", "pixel_sha256"):
            shared = {records[i][key] for i in splits[left]} & {
                records[i][key] for i in splits[right]
            }
            values[key] = len(shared)
        values["connected_components"] = len(
            {components.root(i) for i in splits[left]} & {components.root(i) for i in splits[right]}
        )
        if any(values.values()):
            raise ValueError(f"Split leakage: {left}/{right}: {values}")
        result["overlaps"][f"{left}/{right}"] = values
    ownership = {i: split for split in names for i in splits[split]}
    cross_near = [
        m
        for m in matches
        if m["kind"] == "near_duplicate"
        and m["first"] in ownership
        and m["second"] in ownership
        and ownership[m["first"]] != ownership[m["second"]]
    ]
    if cross_near:
        raise ValueError("Near-duplicate leakage")
    result["cross_split_near_duplicates"] = len(cross_near)
    result["source_match_counts"] = dict(Counter(m["kind"] for m in matches))
    return result


def prepare_clean_split(data_root: Path, output_dir: Path, seed: int = 20260915) -> dict:
    """Create independently versioned manifests and their reproducible audit."""
    paths = sorted(
        p
        for p in data_root.glob("*/*/*")
        if p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        and p.parent.parent.name in {"train", "val", "test"}
    )
    if not paths:
        raise ValueError(f"No images found in {data_root}")
    with ThreadPoolExecutor(max_workers=8) as pool:
        records = list(pool.map(fingerprint, paths))
    components, matches = connect_records(records)
    splits = assign_splits(records, components, seed)
    report = audit_splits(records, splits, components, matches)
    report.update(
        {
            "seed": seed,
            "strategy": "conservative_group_disjoint_v1",
            "validation_fraction_target": 0.2,
            "original_test_preserved": True,
            "identity_rule": "personN across subtypes; IM-N and NORMAL2-IM-N separately",
            "near_duplicate_rule": "phash_hamming<=6 and thumbnail_correlation>=0.995",
            "limitations": [
                "Filename groups are conservative proxies, not verified patient IDs.",
                "Near-duplicate screening cannot prove absence of all acquisition overlap.",
                "Original test was inspected historically; it is a legacy benchmark.",
            ],
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_hashes = {}
    for split, indices in splits.items():
        path = output_dir / f"{split}_manifest.csv"
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=[
                    "filepath",
                    "split",
                    "label",
                    "label_id",
                    "patient_proxy",
                    "group_id",
                    "sha256",
                    "pixel_sha256",
                ],
            )
            writer.writeheader()
            for index in indices:
                record = records[index]
                writer.writerow(
                    {
                        "filepath": record["filepath"],
                        "split": split,
                        "label": record["label"],
                        "label_id": record["label_id"],
                        "patient_proxy": record["patient_proxy"],
                        "group_id": components.root(index),
                        "sha256": record["sha256"],
                        "pixel_sha256": record["pixel_sha256"],
                    }
                )
        manifest_hashes[split] = hashlib.sha256(path.read_bytes()).hexdigest()
    report["manifest_sha256"] = manifest_hashes
    (output_dir / "split_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    serializable = [{k: v for k, v in r.items() if k != "thumbnail"} for r in records]
    (output_dir / "source_inventory.json").write_text(
        json.dumps(serializable, indent=2), encoding="utf-8"
    )
    (output_dir / "duplicate_pairs.json").write_text(
        json.dumps(matches, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    """Build a clean manifest version without changing the raw files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/chest_xray"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/clean_v1"))
    parser.add_argument("--seed", type=int, default=20260915)
    args = parser.parse_args()
    print(json.dumps(prepare_clean_split(args.data_root, args.output_dir, args.seed), indent=2))


if __name__ == "__main__":
    main()
