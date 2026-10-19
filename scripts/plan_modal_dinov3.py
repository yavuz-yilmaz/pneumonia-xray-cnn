"""Offline plan and payload validation; no Modal client, authentication or network."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cloud.dinov3_job import budget_plan, build_payload  # noqa: E402


def main() -> None:
    """Report the cost envelope and optionally validate existing small manifests."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--manifests", type=Path, default=ROOT / "data/processed/clean_v1")
    args = parser.parse_args()
    config = json.loads((ROOT / "configs/modal_dinov3.json").read_text(encoding="utf-8"))
    report = budget_plan(config)
    if args.check:
        payload = build_payload(args.manifests)
        report["upload_payload_bytes"] = len(json.dumps(payload).encode())
        report["uploaded_files"] = ["train_manifest.csv", "val_manifest.csv", "split_audit.json"]
        report["raw_images_uploaded"] = 0
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
