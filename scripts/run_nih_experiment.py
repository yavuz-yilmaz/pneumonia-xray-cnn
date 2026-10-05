"""Run the registered NIH experiment after every source archive passes extraction."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from src.data.download_nih import file_sha256, validate_record

SOURCE_FILES = (
    "configs/nih_download_manifest.json",
    "docs/nih_training_protocol.md",
    "src/data/prepare_nih_split.py",
    "src/data/clean_split.py",
    "src/data/dataset.py",
    "src/data/nih_dataset.py",
    "src/data/image_preparation.py",
    "src/training/train_clean.py",
    "src/training/models.py",
    "src/evaluation/decision.py",
    "src/evaluation/evaluate_nih.py",
)


def extraction_count(extraction: Path, archives: list[dict]) -> int:
    """Count passed extraction audits; refuse contradictory completed artifacts."""
    count = 0
    for archive in archives:
        validate_record(archive)
        path = extraction / archive["name"].removesuffix(".tar.gz") / "image_audit.json"
        if not path.exists():
            continue
        report = json.loads(path.read_bytes())
        if (
            report.get("archive") != archive["name"]
            or report.get("gzip_crc_verified") is not True
            or report.get("all_image_pixels_decoded") is not True
            or report.get("images", 0) < 1
        ):
            raise ValueError("Completed extraction audit failed the source integrity gate")
        count += 1
    return count


def experiment_stages() -> list[tuple[str, list[str]]]:
    """Keep stage order and registered candidate choices explicit and reproducible."""
    common = [
        "--task",
        "nih_report_pneumonia",
        "--manifests",
        "data/processed/nih_pneumonia_v1",
        "--cache-images",
        "--epochs",
        "22",
        "--minimum-recall",
        "0.90",
        "--balanced-sampling",
        "--epoch-samples",
        "16000",
    ]
    return [
        (
            "split",
            ["src.data.prepare_nih_split", "--manifest", "configs/nih_download_manifest.json"],
        ),
        (
            "resnet18",
            [
                "src.training.train_clean",
                *common,
                "--output",
                "models/nih_v1/resnet18_clahe",
                "--model",
                "resnet18",
                "--preprocessing",
                "clahe",
                "--size",
                "224",
                "--batch-size",
                "24",
            ],
        ),
        (
            "efficientnet_v2_s",
            [
                "src.training.train_clean",
                *common,
                "--output",
                "models/nih_v1/efficientnet_v2_s",
                "--model",
                "efficientnet_v2_s",
                "--preprocessing",
                "resize",
                "--size",
                "320",
                "--batch-size",
                "8",
            ],
        ),
        (
            "select",
            [
                "src.evaluation.evaluate_nih",
                "select",
                "--experiments",
                "models/nih_v1/resnet18_clahe",
                "models/nih_v1/efficientnet_v2_s",
            ],
        ),
        ("test", ["src.evaluation.evaluate_nih", "test"]),
    ]


def execute_stage(name: str, arguments: list[str], output: Path) -> None:
    """Stop the dependency chain on failure and retain the entire stage log."""
    command = [sys.executable, "-u", "-m", *arguments]
    with (output / f"{name}.log").open("x", encoding="utf-8") as stream:
        subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)  # noqa: S603


def main() -> None:
    """Wait for verified files, then execute split/train/select/test exactly once."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-for-extraction", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("reports/metrics/nih_run_v1"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(
            "NIH experiment coordinator already exists; inspect it before resuming"
        )
    guarded = [
        Path("data/processed/nih_pneumonia_v1"),
        Path("models/nih_v1/resnet18_clahe"),
        Path("models/nih_v1/efficientnet_v2_s"),
        Path("models/nih_v1/selected"),
        Path("reports/metrics/nih_v1"),
    ]
    if any(path.exists() for path in guarded):
        raise FileExistsError("A registered experiment stage already exists; preserving its state")
    hashes = {name: file_sha256(Path(name)) for name in SOURCE_FILES}
    hashes[__file__] = file_sha256(Path(__file__))
    manifest = json.loads(Path("configs/nih_download_manifest.json").read_bytes())
    archives = manifest["archives"]
    if len(archives) != 12 or len({row["name"] for row in archives}) != 12:
        raise ValueError("Expected all twelve distinct NIH archives")
    args.output.mkdir(parents=True)
    status = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "process_id": os.getpid(),
        "source_sha256": hashes,
        "state": "waiting_for_extraction",
        "completed_stages": [],
        "test_started": False,
    }

    def save() -> None:
        status["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        temporary = args.output / "status.tmp"
        temporary.write_text(json.dumps(status, indent=2), encoding="utf-8")
        temporary.replace(args.output / "status.json")

    save()
    try:
        last_count = -1
        while True:
            count = extraction_count(Path("data/processed/nih_images_v1"), archives)
            if count != last_count:
                status["completed_extraction_archives"] = count
                save()
                print(
                    f"Verified extraction audits: {count}/12; waiting before final split",
                    flush=True,
                )
                last_count = count
            if count == len(archives):
                break
            if not args.wait_for_extraction:
                raise ValueError("Source extraction is incomplete")
            time.sleep(10)
        for name, arguments in experiment_stages():
            if any(file_sha256(Path(path)) != digest for path, digest in hashes.items()):
                raise ValueError("Experiment implementation changed after coordinator registration")
            status["state"] = "running"
            status["stage"] = name
            status["test_started"] = name == "test"
            save()
            print(
                f"Starting registered stage: {name}; log: {args.output / (name + '.log')}",
                flush=True,
            )
            execute_stage(name, arguments, args.output)
            status["completed_stages"].append(name)
            save()
        status["state"] = "complete"
        status["note"] = (
            "Experiment finished; inspect metrics against the user's improvement objective"
        )
        save()
        print("NIH experiment completed; final metric review remains necessary", flush=True)
    except Exception as error:
        status["state"] = "failed"
        status["error"] = f"{type(error).__name__}: {error}"
        save()
        raise


if __name__ == "__main__":
    main()
