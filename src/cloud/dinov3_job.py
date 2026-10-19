"""Small audited payloads, remote-only downloads and conservative spend reservations."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import re
import tarfile
import time
import zipfile
from pathlib import Path, PurePosixPath

MAX_BUDGET_USD = 28.0


def modal_resource_name(variable: str, default: str) -> str:
    """Use a stable, project-neutral Modal resource name with an optional local override."""
    return os.environ.get(variable, default)


APP_NAME = modal_resource_name("MODAL_DINOV3_APP_NAME", "pneumonia-dinov3")
VOLUME_NAME = modal_resource_name("MODAL_DINOV3_VOLUME_NAME", "pneumonia-dinov3-data")
PILOT_TIMING_SCHEMA = 3
SPLITS = ("train", "val")
RECIPE_KEYS = (
    "model_id",
    "model_revision",
    "image_size",
    "batch_size",
    "epochs",
    "warmup_epochs",
    "tail_blocks",
    "head_lr",
    "backbone_lr",
    "weight_decay",
    "patience",
    "minimum_recall",
    "seed",
    "num_workers",
)


def recipe(config: dict) -> dict:
    """Separate reproducible training choices from adjustable wall-clock limits."""
    return {key: config[key] for key in RECIPE_KEYS}


def check_model_access(config: dict, token: str) -> dict:
    """Check the gated pinned file with HEAD, without downloads or raw error text."""
    from huggingface_hub import get_hf_file_metadata, hf_hub_url
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError
    from requests.exceptions import RequestException

    url = hf_hub_url(config["model_id"], "config.json", revision=config["model_revision"])
    try:
        get_hf_file_metadata(url, token=token, timeout=15)
    except GatedRepoError:
        status = "gated_model"
        message = "Request DINOv3 access and use a token from the approved Hugging Face account."
    except HfHubHTTPError as error:
        code = error.response.status_code if error.response is not None else None
        status = "invalid_token" if code == 401 else "http_error"
        message = f"Hugging Face HTTP {code}; check model access and the Modal HF_TOKEN Secret."
    except RequestException:
        status = "network_error"
        message = "Cannot reach Hugging Face; retry the access check before running the pilot."
    else:
        return {"access_granted": True, "status": "ready"}
    # Never serialize the HTTP exception: it can contain a large server response.
    return {"access_granted": False, "status": status, "message": message}


def digest(content: bytes) -> str:
    """Identify source bytes without exposing file contents."""
    return hashlib.sha256(content).hexdigest()


def write_json(path: Path, value: dict | list) -> None:
    """Replace an artifact atomically on the same filesystem."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def validate_config(config: dict) -> None:
    """Fail closed on a larger budget, invalid rates or an unsupported experiment."""
    for name in (
        "budget_usd",
        "reserve_usd",
        "gpu_usd_per_second",
        "cpu_usd_per_core_second",
        "memory_usd_per_gib_second",
        "coordinator_allowance_usd",
        "head_lr",
        "backbone_lr",
    ):
        value = config[name]
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"Invalid positive finite setting: {name}")
    if config["budget_usd"] > MAX_BUDGET_USD or config["reserve_usd"] < 4:
        raise ValueError("Budget must be <= $28 with at least $4 held in reserve")
    if config["reserve_usd"] >= config["budget_usd"]:
        raise ValueError("Reserve consumes the whole budget")
    for name in (
        "prepare_timeout_seconds",
        "pilot_timeout_seconds",
        "train_timeout_seconds",
        "startup_allowance_seconds",
        "idle_allowance_seconds",
        "cpu_cores",
        "memory_gib",
        "image_size",
        "batch_size",
        "epochs",
        "tail_blocks",
        "patience",
    ):
        if type(config[name]) is not int or config[name] <= 0:
            raise ValueError(f"Invalid positive integer setting: {name}")
    if any(config[f"{stage}_timeout_seconds"] > 21600 for stage in ("prepare", "pilot", "train")):
        raise ValueError("Stage timeouts must be <= six hours")
    if config["gpu"] != "A100-80GB" or config["cpu_cores"] != 2 or config["memory_gib"] != 16:
        raise ValueError("This costed experiment requires A100-80GB, 2 cores, 16 GiB RAM")
    if config["image_size"] % 16 or not 0 < config["minimum_recall"] <= 1:
        raise ValueError("Require a multiple-of-16 image size and valid recall floor")
    if not 0 <= config["warmup_epochs"] < config["epochs"]:
        raise ValueError("Warmup must be shorter than training")
    if config["model_id"] != "facebook/dinov3-vitl16-pretrain-lvd1689m":
        raise ValueError("The registered budget recipe supports ViT-L/16 only")
    if re.fullmatch(r"[a-f0-9]{40}", config["model_revision"]) is None:
        raise ValueError("Pin a full Hugging Face commit revision")
    if not config["dataset_url"].startswith("https://www.kaggle.com/api/v1/datasets/download/"):
        raise ValueError("Use the registered HTTPS Kaggle dataset endpoint")


def stage_cost(config: dict, stage: str) -> float:
    """Reserve timeout + startup + idle; include requested CPU and host RAM."""
    validate_config(config)
    if stage.startswith("external_"):
        from src.cloud.dinov3_external import external_stage_cost

        return external_stage_cost(config, stage)
    if stage == "coordinator":
        return float(config["coordinator_allowance_usd"])
    if stage not in {"prepare", "pilot", "train"}:
        raise ValueError("Unknown funded stage")
    seconds = (
        config[f"{stage}_timeout_seconds"]
        + config["startup_allowance_seconds"]
        + config["idle_allowance_seconds"]
    )
    rate = (
        config["cpu_cores"] * config["cpu_usd_per_core_second"]
        + config["memory_gib"] * config["memory_usd_per_gib_second"]
    )
    if stage != "prepare":
        rate += config["gpu_usd_per_second"]
    return math.ceil(seconds * rate * 10000) / 10000


def budget_plan(config: dict) -> dict:
    """Describe an initial prepare/pilot/train run without calling Modal."""
    costs = {
        stage: stage_cost(config, stage) for stage in ("coordinator", "prepare", "pilot", "train")
    }
    total = round(sum(costs.values()), 4)
    allowance = config["budget_usd"] - config["reserve_usd"]
    if total > allowance:
        raise ValueError(f"Planned reservations ${total:.2f} exceed allowance ${allowance:.2f}")
    return {
        "budget_usd": config["budget_usd"],
        "unallocated_reserve_usd": config["reserve_usd"],
        "stage_reservations_usd": costs,
        "total_reserved_usd": total,
        "remaining_run_allowance_usd": round(allowance - total, 4),
        "invoice_cap": "Set a $28 workspace budget in Modal; estimates are not an invoice cap",
    }


def reserve_cost(path: Path, config: dict, stage: str, job_id: str) -> dict:
    """Append worst-case usage before dispatch; failures never refund reservations.

    Only the deployed single-container coordinator may write this ledger. A
    separate ephemeral app per launch would not serialize writers across runs.
    """
    cost = stage_cost(config, stage)
    ceiling = config["budget_usd"] - config["reserve_usd"]
    ledger = (
        json.loads(path.read_text(encoding="utf-8"))
        if path.exists()
        else {"ceiling_usd": ceiling, "entries": [], "reserved_usd": 0.0}
    )
    if ledger["ceiling_usd"] != ceiling:
        raise ValueError("Existing ledger budget cannot be changed or reset")
    used = sum(entry["reserved_usd"] for entry in ledger["entries"])
    if not math.isfinite(used) or abs(used - ledger["reserved_usd"]) > 1e-5:
        raise ValueError("Budget ledger is inconsistent")
    if used + cost > ceiling + 1e-8:
        raise ValueError(f"Budget exhausted: ${used:.2f} reserved; next stage needs ${cost:.2f}")
    ledger["entries"].append(
        {"stage": stage, "job_id": job_id, "reserved_usd": cost, "timestamp_unix": time.time()}
    )
    ledger["reserved_usd"] = round(used + cost, 4)
    write_json(path, ledger)
    return ledger


def image_key(filepath: str) -> str:
    """Map Windows/POSIX source paths to the original split/label/basename."""
    parts = PurePosixPath(filepath.replace("\\", "/")).parts
    if len(parts) < 3:
        raise ValueError("Manifest image path has no source split/label")
    split, label, filename = parts[-3:]
    if split not in {"train", "val"} or label not in {"NORMAL", "PNEUMONIA"}:
        raise ValueError("Development payload must not reference test images")
    if filename in {"", ".", ".."} or ":" in filename:
        raise ValueError("Invalid image basename")
    return "/".join((split, label, filename))


def build_payload(manifests: Path) -> dict:
    """Transfer only train/val CSVs and the existing audit, never images or test CSV."""
    audit = json.loads((manifests / "split_audit.json").read_text(encoding="utf-8"))
    if (
        audit["cross_split_near_duplicates"]
        or not audit["overlaps"]
        or any(any(v.values()) for v in audit["overlaps"].values())
    ):
        raise ValueError("Require a nonempty, zero-overlap source audit")
    sources = {}
    for split in SPLITS:
        raw = (manifests / f"{split}_manifest.csv").read_bytes()
        if digest(raw) != audit["manifest_sha256"][split]:
            raise ValueError(f"Source manifest changed after audit: {split}")
        sources[split] = raw.decode("utf-8")
    payload = {"audit": audit, "manifests": sources}
    development_rows(payload)
    return payload


def development_rows(payload: dict) -> dict[str, list[dict]]:
    """Check labels, hashes, row counts and grouped development separation."""
    audit = payload["audit"]
    if (
        audit["cross_split_near_duplicates"]
        or not audit["overlaps"]
        or any(any(v.values()) for v in audit["overlaps"].values())
    ):
        raise ValueError("Source audit reports leakage or lacks overlap checks")
    if set(payload["manifests"]) != set(SPLITS):
        raise ValueError("Payload must contain only train and val")
    result = {}
    for split in SPLITS:
        text = payload["manifests"][split]
        if digest(text.encode("utf-8")) != audit["manifest_sha256"][split]:
            raise ValueError("Payload manifest hash does not match source audit")
        rows = list(csv.DictReader(io.StringIO(text)))
        if not rows or len(rows) != audit["counts"][split]["images"]:
            raise ValueError("Source audit row count mismatch")
        for row in rows:
            image_key(row["filepath"])
            if row["split"] != split or row["label_id"] != str(
                {"NORMAL": 0, "PNEUMONIA": 1}[row["label"]]
            ):
                raise ValueError("Manifest label or split mismatch")
            for key in ("sha256", "pixel_sha256"):
                if re.fullmatch(r"[a-f0-9]{64}", row[key]) is None:
                    raise ValueError("Invalid image fingerprint")
        if len({image_key(r["filepath"]) for r in rows}) != len(rows):
            raise ValueError("Duplicate source image key")
        result[split] = rows
    for key in ("patient_proxy", "group_id", "sha256", "pixel_sha256"):
        if {r[key] for r in result["train"]} & {r[key] for r in result["val"]}:
            raise ValueError(f"Development overlap: {key}")
    return result


def extract_development(archive: Path, payload: dict, output: Path) -> dict:
    """Extract only exact audited train/val members; never materialize test images."""
    rows = development_rows(payload)
    wanted = {image_key(row["filepath"]): row for split in SPLITS for row in rows[split]}
    with zipfile.ZipFile(archive) as source:
        members: dict[str, list[zipfile.ZipInfo]] = {}
        for member in source.infolist():
            path = PurePosixPath(member.filename)
            if member.is_dir() or "__MACOSX" in path.parts or len(path.parts) < 3:
                continue
            key = "/".join(path.parts[-3:])
            if key in wanted:
                if member.file_size > 30_000_000:
                    raise ValueError(
                        f"Oversized dataset member: {member.filename} ({member.file_size} bytes)"
                    )
                members.setdefault(key, []).append(member)
        if set(members) != set(wanted):
            raise ValueError(f"Archive lacks {len(set(wanted) - set(members))} audited images")
        for key, row in wanted.items():
            # Kaggle mirrors images under chest_xray/ and chest_xray/chest_xray/.
            # Accept repeated paths only when every copy matches the audited bytes.
            for member in members[key]:
                content = source.read(member)
                if digest(content) != row["sha256"]:
                    raise ValueError(
                        f"Downloaded image differs from original audit: {key} ({member.filename})"
                    )
            target = output / "images" / key
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
    # Rebase paths only, retain labels/group membership and preserve the original audit.
    audit = json.loads(json.dumps(payload["audit"]))
    audit["source_manifest_sha256"] = dict(audit["manifest_sha256"])
    for split in SPLITS:
        path = output / f"{split}_manifest.csv"
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[split][0]))
            writer.writeheader()
            for row in rows[split]:
                writer.writerow(
                    {**row, "filepath": str(output / "images" / image_key(row["filepath"]))}
                )
        audit["manifest_sha256"][split] = digest(path.read_bytes())
    audit["relocation"] = "Paths changed; exact source bytes and original split membership retained"
    write_json(output / "split_audit.json", audit)
    return audit


def file_digest(path: Path) -> str:
    """Hash a large cloud artifact without loading it into host memory."""
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(2 * 1024 * 1024):
            checksum.update(chunk)
    return checksum.hexdigest()


def create_development_bundle(archive: Path, payload: dict, cohort: Path) -> str:
    """Package audited development bytes as one sequential GPU staging artifact."""
    rows = development_rows(payload)
    wanted = {image_key(row["filepath"]): row for split in SPLITS for row in rows[split]}
    destination = cohort / "development.tar"
    temporary = destination.with_suffix(".tmp")
    with zipfile.ZipFile(archive) as source, tarfile.open(temporary, "w") as bundle:
        for name in ("train_manifest.csv", "val_manifest.csv", "split_audit.json"):
            content = (cohort / name).read_bytes()
            header = tarfile.TarInfo(name)
            header.size = len(content)
            bundle.addfile(header, io.BytesIO(content))
        added = set()
        for member in source.infolist():
            parts = PurePosixPath(member.filename).parts
            key = "/".join(parts[-3:])
            if member.is_dir() or "__MACOSX" in parts or key not in wanted or key in added:
                continue
            if member.file_size > 30_000_000:
                raise ValueError("Oversized bundle source image")
            content = source.read(member)
            if digest(content) != wanted[key]["sha256"]:
                raise ValueError(f"Bundle image differs from original audit: {key}")
            header = tarfile.TarInfo(f"images/{key}")
            header.size = len(content)
            bundle.addfile(header, io.BytesIO(content))
            added.add(key)
        if added != set(wanted):
            raise ValueError("Bundle lacks audited development images")
    temporary.replace(destination)
    return file_digest(destination)


def copy_verified(source: Path, destination: Path, expected_sha256: str) -> None:
    """Stream a Volume file to the container disk and verify the exact bytes."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    checksum = hashlib.sha256()
    try:
        with source.open("rb") as incoming, temporary.open("wb") as outgoing:
            while chunk := incoming.read(2 * 1024 * 1024):
                checksum.update(chunk)
                outgoing.write(chunk)
        if checksum.hexdigest() != expected_sha256:
            raise ValueError(f"Cloud artifact changed after preparation: {source.name}")
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def localize_inputs(cohort: Path, weights: Path, destination: Path) -> tuple[Path, Path]:
    """Stage bulk files on the GPU container's disk; retain all development checks."""
    prepared = json.loads((cohort / "prepared.json").read_text(encoding="utf-8"))
    bundle_path = destination / "development.tar"
    copy_verified(cohort / "development.tar", bundle_path, prepared["bundle_sha256"])
    extracted = set()
    metadata = {"train_manifest.csv", "val_manifest.csv", "split_audit.json"}
    with tarfile.open(bundle_path, "r|") as bundle:
        for member in bundle:
            if not member.isfile() or member.name in extracted or member.size > 30_000_000:
                raise ValueError("Invalid development bundle member")
            if member.name not in metadata:
                key = image_key(member.name)
                if member.name != f"images/{key}":
                    raise ValueError("Invalid development bundle image path")
            with bundle.extractfile(member) as incoming:
                target = destination / member.name
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("wb") as outgoing:
                    while chunk := incoming.read(2 * 1024 * 1024):
                        outgoing.write(chunk)
            extracted.add(member.name)
    if not metadata <= extracted:
        raise ValueError("Development bundle lacks audited manifests")
    audit = json.loads((destination / "split_audit.json").read_text(encoding="utf-8"))
    payload = {
        "audit": audit,
        "manifests": {
            split: (destination / f"{split}_manifest.csv").read_bytes().decode("utf-8")
            for split in SPLITS
        },
    }
    rows = development_rows(payload)
    wanted = {f"images/{image_key(row['filepath'])}" for split in SPLITS for row in rows[split]}
    if extracted != metadata | wanted:
        raise ValueError("Development bundle image set differs from the audited manifests")
    for split in SPLITS:
        path = destination / f"{split}_manifest.csv"
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[split][0]))
            writer.writeheader()
            for row in rows[split]:
                writer.writerow(
                    {**row, "filepath": str(destination / "images" / image_key(row["filepath"]))}
                )
        audit["manifest_sha256"][split] = file_digest(path)
    write_json(destination / "split_audit.json", audit)
    local_weights = destination / "weights"
    for name, checksum in prepared["weight_files_sha256"].items():
        if PurePosixPath(name).name != name or not name.endswith((".json", ".safetensors")):
            raise ValueError("Invalid prepared weight filename")
        copy_verified(weights / name, local_weights / name, checksum)
    return destination, local_weights


def download_dataset(url: str, destination: Path, deadline: float) -> str:
    """Stream a public/archive-authorized download inside Modal, with no local upload."""
    import requests

    auth = None
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        auth = (os.environ["KAGGLE_USERNAME"], os.environ["KAGGLE_KEY"])
    temporary = destination.with_suffix(".partial")
    try:
        with requests.get(url, auth=auth, stream=True, timeout=(30, 60)) as response:
            if response.status_code != 200:
                raise RuntimeError(
                    f"Dataset access failed (HTTP {response.status_code}); check Kaggle access"
                )
            if "text/html" in response.headers.get("Content-Type", ""):
                raise RuntimeError("Dataset endpoint returned HTML; configure Kaggle access")
            checksum = hashlib.sha256()
            size = 0
            with temporary.open("wb") as stream:
                for chunk in response.iter_content(1024 * 1024):
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Dataset preparation time limit reached")
                    size += len(chunk)
                    if size > 8_000_000_000:
                        raise ValueError("Unexpected archive size; refuse download above 8 GB")
                    stream.write(chunk)
                    checksum.update(chunk)
        if not zipfile.is_zipfile(temporary):
            raise ValueError("Dataset response is not a ZIP archive")
        temporary.replace(destination)
        return checksum.hexdigest()
    finally:
        if temporary.exists():
            temporary.unlink()
