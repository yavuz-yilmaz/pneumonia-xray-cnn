"""Locked external evaluation; public downloads and all image inference stay in Modal."""

from __future__ import annotations

import hashlib
import io
import json
import math
import re
import tarfile
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

from src.cloud.dinov3_job import digest, file_digest, write_json

JOB_ID = "1b73b1facc83fda1"
THRESHOLD = 0.6127732396125793
STARTUP_SECONDS = 180
TIMEOUT_SECONDS = 600
IDLE_SECONDS = 10
MONTGOMERY_BASE = (
    "https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/"
    "Montgomery-County-CXR-Set/MontgomerySet/CXR_png/"
)


def external_stage_cost(config: dict, stage: str) -> float:
    """Cost fixed resource limits, including full coordinator lifetime and startup."""
    if stage == "external_coordinator":
        seconds = 28800 + 600 + IDLE_SECONDS
        rate = 0.125 * config["cpu_usd_per_core_second"]
        rate += 0.5 * config["memory_usd_per_gib_second"]
    elif stage in {"external_prepare", "external_test"}:
        seconds = TIMEOUT_SECONDS + STARTUP_SECONDS + IDLE_SECONDS
        rate = config["cpu_usd_per_core_second"] + 8 * config["memory_usd_per_gib_second"]
        if stage == "external_test":
            rate += 0.000222  # Modal L4 list price checked 2026-10-07.
    else:
        raise ValueError("Unknown external stage")
    return math.ceil(seconds * rate * 10000) / 10000


def build_external_payload(root: Path) -> dict:
    """Transfer public metadata/fingerprints only; no local image or checkpoint bytes."""
    metadata_path = root / "reports/metrics/modal_dinov3_external_v1/ssmu2026.txt"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
    if str(metadata["id"]) != "18241541" or metadata["metadata"]["license"]["id"] != "cc-by-4.0":
        raise ValueError("Unexpected SSMU record or license")
    files = sorted(
        [
            {
                "name": f["key"],
                "size": f["size"],
                "checksum": f["checksum"],
                "url": f["links"]["self"],
            }
            for f in metadata["files"]
            if f["key"].endswith("_pa.png")
        ],
        key=lambda f: f["name"],
    )
    if len(files) != 50:
        raise ValueError("Expected 50 frontal SSMU images")
    references, source_hashes = [], {}
    for source, relative in (
        ("kermany_all_splits", "data/processed/clean_v1/source_inventory.json"),
        ("ssmu2021", "data/processed/ssmu_images_v1/images.json"),
    ):
        path = root / relative
        source_hashes[source] = file_digest(path)
        for index, record in enumerate(json.loads(path.read_bytes())):
            references.append(
                {
                    "id": f"{source}:{index}",
                    "sha256": record["sha256"],
                    "pixel_sha256": record["pixel_sha256"],
                    "phash": record["phash"],
                }
            )
    return {
        "protocol": {
            "schema": 1,
            "job_id": JOB_ID,
            "best_epoch": 4,
            "threshold": THRESHOLD,
            "image_size": 512,
            "label_mapping": {"NORMAL": 0, "PNEUMONIA": 1},
            "selection": "RSUA validated original BMP pneumonia/normal; SSMU2026 part1 all _pa.png",
            "rsua": {
                "doi": "10.17632/2jg8vfdmpm.1",
                "license": "CC BY 4.0",
                "url": "https://data.mendeley.com/public-files/datasets/2jg8vfdmpm/files/"
                "f7dbac97-e7f0-4185-8cdb-ec15f130a3f8/file_downloaded",
                "sha256": "ce2e1215e4ca1051686ac0931f9056ee8c320d2f29feefd7ce6f4bb0c4c5177e",
                "expected_counts": {"NORMAL": 32, "PNEUMONIA": 53},
            },
            "ssmu2026": {
                "doi": "10.5281/zenodo.18241541",
                "license": "CC BY 4.0",
                "files": files,
                "task": "pneumonia sensitivity only",
            },
            "reference_sha256": source_hashes,
            "overlap_rule": (
                "exclude byte/pixel matches or pHash Hamming <=6; conservative near screening"
            ),
            "inference": (
                "RGB resize512 ImageNet normalization; eval; bfloat16; "
                "no training or threshold search"
            ),
            "patient_independence": "not verified; filenames are not verified patient identifiers",
            "ssmu_institution_unseen": False,
        },
        "references": references,
    }


def external_id(payload: dict) -> str:
    """Pin metadata, cohort rule and all overlap references before accessing scores."""
    if payload["protocol"]["job_id"] != JOB_ID or payload["protocol"]["threshold"] != THRESHOLD:
        raise ValueError("External checkpoint/threshold changed")
    return digest(json.dumps(payload, sort_keys=True).encode())[:16]


def fetch(url: str, deadline: float, limit: int) -> bytes:
    """One bounded public request, with no local downloads, retries or credentials."""
    import requests

    if not url.startswith(("https://data.mendeley.com/", "https://zenodo.org/", MONTGOMERY_BASE)):
        raise ValueError("Unregistered publisher endpoint")
    chunks, size = [], 0
    with requests.get(url, stream=True, timeout=(15, 30)) as response:
        response.raise_for_status()
        for chunk in response.iter_content(1024 * 1024):
            if time.monotonic() > deadline:
                raise TimeoutError("External download deadline")
            size += len(chunk)
            if size > limit:
                raise ValueError("External file exceeds limit")
            chunks.append(chunk)
    return b"".join(chunks)


def prepare_external(payload: dict, output: Path) -> dict:
    """Download once, pin source hashes and package data for sequential Volume reads."""
    from PIL import Image

    started = time.monotonic()
    deadline = started + TIMEOUT_SECONDS - 75
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "protocol.json", payload["protocol"])
    records, errors, archive_members = [], {}, []
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as bundle:

        def add(data: bytes, dataset: str, name: str, label: int | None) -> None:
            with Image.open(io.BytesIO(data)) as image:
                image.verify()
            key = f"{dataset}/{len(records):04d}{PurePosixPath(name).suffix.lower()}"
            entry = tarfile.TarInfo(key)
            entry.size = len(data)
            bundle.addfile(entry, io.BytesIO(data))
            records.append(
                {
                    "dataset": dataset,
                    "source_name": name,
                    "key": key,
                    "label": label,
                    "sha256": digest(data),
                }
            )

        try:
            source = payload["protocol"]["rsua"]
            data = fetch(source["url"], deadline, 40 * 1024 * 1024)
            if digest(data) != source["sha256"]:
                raise ValueError("RSUA publisher checksum mismatch")
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                archive_members = [x.filename for x in archive.infolist() if not x.is_dir()]
                for member in archive.infolist():
                    if not member.is_dir() and member.filename.lower().endswith(".bmp"):
                        if member.file_size > 4 * 1024 * 1024:
                            raise ValueError("Oversized RSUA member")
                        # Layout is reviewed and frozen before the separate inference action.
                        add(archive.read(member), "rsua", member.filename, None)
        except Exception as error:
            errors["rsua"] = type(error).__name__ + ": " + str(error)[:180]

        files = payload["protocol"]["ssmu2026"]["files"]

        def download_file(item: dict) -> tuple[dict, bytes]:
            content = fetch(item["url"], deadline, item["size"])
            expected = item["checksum"].removeprefix("md5:")
            if (
                len(content) != item["size"]
                or hashlib.md5(content, usedforsecurity=False).hexdigest() != expected
            ):
                raise ValueError("SSMU publisher checksum mismatch")
            return item, content

        try:
            with ThreadPoolExecutor(max_workers=6) as executor:
                for item, content in executor.map(download_file, files):
                    add(content, "ssmu2026", item["name"], 1)
        except Exception as error:
            errors["ssmu2026"] = type(error).__name__ + ": " + str(error)[:180]
    content = buffer.getvalue()
    (output / "images.tar").write_bytes(content)
    report = {
        "external_id": external_id(payload),
        "records": records,
        "errors": errors,
        "archive_members": archive_members,
        "bundle_sha256": digest(content),
        "elapsed_seconds": time.monotonic() - started,
        "predictions_computed": False,
    }
    write_json(output / "prepared.json", report)
    return report


def metrics(labels: list[int], probabilities: list[float], threshold: float) -> dict:
    """Do not turn positive-only cohorts into misleading binary performance claims."""
    import numpy as np
    from sklearn.metrics import average_precision_score, roc_auc_score

    y, p = np.asarray(labels), np.asarray(probabilities)
    if (
        not len(y)
        or y.shape != p.shape
        or not np.isfinite(p).all()
        or not np.isin(y, [0, 1]).all()
        or ((p < 0) | (p > 1)).any()
    ):
        raise ValueError("Invalid external predictions")
    predicted = p >= threshold
    tp = int(((y == 1) & predicted).sum())
    fn = int(((y == 1) & ~predicted).sum())
    tn = int(((y == 0) & ~predicted).sum())
    fp = int(((y == 0) & predicted).sum())

    def rate_ci(successes: int, total: int) -> dict | None:
        if not total:
            return None
        rate, z = successes / total, 1.959963984540054
        denominator = 1 + z * z / total
        center = (rate + z * z / (2 * total)) / denominator
        margin = z * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total**2)) / denominator
        return {"value": rate, "wilson95": [center - margin, center + margin]}

    both = tp + fn > 0 and tn + fp > 0
    return {
        "sample_count": len(y),
        "threshold": threshold,
        "tp": tp,
        "fn": fn,
        "tn": tn,
        "fp": fp,
        "recall": rate_ci(tp, tp + fn),
        "specificity": rate_ci(tn, tn + fp),
        "accuracy": rate_ci(tp + tn, len(y)) if both else None,
        "precision": tp / (tp + fp) if both and tp + fp else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if both else None,
        "roc_auc": float(roc_auc_score(y, p)) if both else None,
        "average_precision": float(average_precision_score(y, p)) if both else None,
        "ci_unit": "image; patient independence unverified",
    }


def display_image(content: bytes, dataset: str) -> tuple:
    """Use a declared 12-bit display range for original Montgomery integer PNGs."""
    import numpy as np
    from PIL import Image, ImageOps

    with Image.open(io.BytesIO(content)) as source:
        source = ImageOps.exif_transpose(source)
        pixels = np.asarray(source)
        audit = {
            "source_mode": source.mode,
            "minimum": int(pixels.min()),
            "maximum": int(pixels.max()),
            "display_conversion": "native RGB conversion",
        }
        if dataset == "montgomery_normal" and source.mode in {"I", "I;16", "I;16B", "I;16L"}:
            if pixels.min() < 0 or pixels.max() > 4095:
                raise ValueError("Montgomery integer PNG exceeds publisher's 12-bit range")
            pixels = np.rint(pixels.astype(np.float32) * (255 / 4095)).astype(np.uint8)
            audit["display_conversion"] = "fixed 12-bit 0..4095 to 8-bit 0..255"
            return Image.fromarray(pixels).convert("RGB"), audit
        return source.convert("RGB"), audit


def validate_selection(payload: dict, prepared: dict, selection: dict) -> list[dict]:
    """Require complete pre-score cohorts, normal-only controls and publisher URLs."""
    if selection["external_id"] != external_id(payload):
        raise ValueError("Selection/source protocol changed")
    candidates = {r["key"]: r for r in prepared["records"]}
    records = []
    for chosen in selection["records"]:
        row = candidates[chosen["key"]]
        if row["dataset"] == "rsua":
            raise ValueError("RSUA archive member layout has not been reviewed; cannot score")
        if chosen["label"] not in (0, 1):
            raise ValueError("External label must be normal/pneumonia")
        if row["dataset"] == "ssmu2026" and chosen["label"] != 1:
            raise ValueError("SSMU is positive only")
        records.append({**row, "label": chosen["label"]})
    ssmu = [r for r in records if r["dataset"] == "ssmu2026"]
    expected = {f["name"] for f in payload["protocol"]["ssmu2026"]["files"]}
    if {r["source_name"] for r in ssmu} != expected or "ssmu2026" in prepared["errors"]:
        raise ValueError("Incomplete SSMU cohort; no partial-source metrics")
    if not records or len({r["key"] for r in records}) != len(records):
        raise ValueError("Empty or duplicate external selection")
    supplement = selection.get("supplement")
    if supplement:
        if supplement["dataset"] != "montgomery_normal" or supplement["task"] != "specificity only":
            raise ValueError("Supplement must be normal-only Montgomery controls")
        files = supplement["files"]
        if len(files) != 80 or len({f["name"] for f in files}) != 80:
            raise ValueError("Expected all 80 Montgomery normal controls")
        for item in files:
            if (
                not re.fullmatch(r"MCUCXR_[0-9]+_0\.png", item["name"])
                or item["url"] != MONTGOMERY_BASE + item["name"]
                or not 0 < item["size"] <= 10 * 1024 * 1024
            ):
                raise ValueError("Invalid Montgomery normal file/URL/size")
    return records


def screen(
    records: list[dict], blobs: dict[str, bytes], references: list[dict]
) -> tuple[list[dict], list[dict]]:
    """Exclude any suspect overlap before inference, including within-source duplicates."""
    from src.data.extract_rsna_cohort import image_signature

    kept, excluded = [], []
    known_sha = {r["sha256"] for r in references}
    known_pixels = {r["pixel_sha256"] for r in references}
    known_phash = [r["phash"] for r in references]
    for record in records:
        image, display_audit = display_image(blobs[record["key"]], record.get("dataset", ""))
        pixel_hash, phash, _ = image_signature(image)
        reason = None
        if record["sha256"] in known_sha or pixel_hash in known_pixels:
            reason = "exact_overlap"
        elif any((phash ^ other).bit_count() <= 6 for other in known_phash):
            reason = "conservative_near_overlap"
        if reason:
            excluded.append({**record, "reason": reason})
        else:
            kept.append(
                {
                    **record,
                    "pixel_sha256": pixel_hash,
                    "phash": phash,
                    "display_audit": display_audit,
                }
            )
        known_sha.add(record["sha256"])
        known_pixels.add(pixel_hash)
        known_phash.append(phash)
    return kept, excluded


def evaluate_external(payload: dict, selection: dict, root: Path, output: Path) -> dict:
    """Run the epoch4 checkpoint once; prohibit model/threshold or post-score cohort edits."""
    import torch
    from torchvision import transforms

    from src.training.dinov3_model import DINOv3Classifier

    started = time.monotonic()
    torch.set_num_threads(1)
    prepared = json.loads((output / "prepared.json").read_bytes())
    if prepared["external_id"] != external_id(payload):
        raise ValueError("Prepared source protocol changed")
    records = validate_selection(payload, prepared, selection)
    if (output / "predictions.json").exists():
        raise ValueError("External evaluation already consumed; do not retune")
    bundle = (output / "images.tar").read_bytes()
    if digest(bundle) != prepared["bundle_sha256"]:
        raise ValueError("External image bundle changed")
    with tarfile.open(fileobj=io.BytesIO(bundle)) as archive:
        blobs = {r["key"]: archive.extractfile(r["key"]).read() for r in records}
    if any(digest(blobs[r["key"]]) != r["sha256"] for r in records):
        raise ValueError("External image bytes changed")
    write_json(output / "protocol_amendment.json", selection)
    supplement = selection.get("supplement")
    source_errors = dict(prepared["errors"])
    if supplement:

        def download_normal(item: dict) -> tuple[dict, bytes]:
            data = fetch(item["url"], started + TIMEOUT_SECONDS - 180, item["size"])
            if len(data) != item["size"]:
                raise ValueError("Montgomery publisher size mismatch")
            return item, data

        try:
            with ThreadPoolExecutor(max_workers=6) as executor:
                normals = list(executor.map(download_normal, supplement["files"]))
        except Exception as error:
            # Never score a selectively downloaded subset from this source.
            source_errors["montgomery_normal"] = type(error).__name__ + ": " + str(error)[:180]
        else:
            for item, data in normals:
                key = "montgomery_normal/" + item["name"]
                blobs[key] = data
                records.append(
                    {
                        "key": key,
                        "source_name": item["name"],
                        "dataset": "montgomery_normal",
                        "label": 0,
                        "sha256": digest(data),
                        "url": item["url"],
                    }
                )
    kept, excluded = screen(records, blobs, payload["references"])
    local_checkpoint = Path("/tmp/dinov3-external-best.pt")  # noqa: S108
    from src.cloud.dinov3_job import copy_verified

    checkpoint_path = root / "runs" / JOB_ID / "best_model.pt"
    checkpoint_sha = file_digest(checkpoint_path)
    copy_verified(checkpoint_path, local_checkpoint, checkpoint_sha)
    frozen = {
        **payload["protocol"],
        "pre_prediction_amendment": selection,
        "checkpoint_sha256": checkpoint_sha,
        "selection": records,
        "retained": kept,
        "excluded": excluded,
    }
    write_json(output / "locked_inference.json", frozen)
    saved = torch.load(local_checkpoint, map_location="cpu", weights_only=False)
    if (
        saved["format"] != "dinov3_binary_v1"
        or saved["best_epoch"] != 4
        or saved["decision_threshold"] != THRESHOLD
        or saved["image_size"] != 512
        or saved["label_mapping"] != {"NORMAL": 0, "PNEUMONIA": 1}
    ):
        raise ValueError("Checkpoint does not match frozen protocol")
    model = DINOv3Classifier.from_config(saved["backbone_config"])
    model.load_state_dict(saved["model_state_dict"], strict=True)
    del saved["model_state_dict"]
    model.to("cuda").eval().requires_grad_(False)
    normalization = saved["normalization"]
    transform = transforms.Compose(
        [
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(normalization["mean"], normalization["std"]),
        ]
    )
    predictions = []
    for start in range(0, len(kept), 4):
        if time.monotonic() - started > TIMEOUT_SECONDS - 45:
            raise TimeoutError("External inference deadline; no partial metrics")
        batch = kept[start : start + 4]
        images = []
        for row in batch:
            image, _ = display_image(blobs[row["key"]], row["dataset"])
            images.append(transform(image))
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(torch.stack(images).to("cuda"))
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Nonfinite external logits")
        scores = logits.float().softmax(1)[:, 1].cpu().tolist()
        predictions.extend(
            {**r, "probability": score, "predicted_label": int(score >= THRESHOLD)}
            for r, score in zip(batch, scores, strict=True)
        )
        print(
            json.dumps(
                {"stage": "external_inference", "completed": len(predictions), "total": len(kept)}
            ),
            flush=True,
        )
    result = {
        "external_id": external_id(payload),
        "checkpoint_sha256": checkpoint_sha,
        "threshold": THRESHOLD,
        "excluded": excluded,
        "source_errors": source_errors,
        "elapsed_seconds": time.monotonic() - started,
        "gpu": torch.cuda.get_device_name(),
        "datasets": {},
    }
    for dataset in sorted({r["dataset"] for r in predictions}):
        subset = [r for r in predictions if r["dataset"] == dataset]
        result["datasets"][dataset] = metrics(
            [r["label"] for r in subset], [r["probability"] for r in subset], THRESHOLD
        )
    write_json(output / "predictions.json", predictions)
    write_json(output / "result.json", result)
    return result
