"""Prepare and train a pinned DINOv3 experiment on Modal within a $28 budget.

Use the standalone offline plan first. Deploy once, then run actions against that same
deployed coordinator: its single container serializes persistent reservations.
Only code and audited train/val CSVs are transferred, never local images/weights.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import modal
from src.cloud.dinov3_job import (
    APP_NAME,
    PILOT_TIMING_SCHEMA,
    VOLUME_NAME,
    budget_plan,
    build_payload,
    check_model_access,
    create_development_bundle,
    digest,
    download_dataset,
    extract_development,
    file_digest,
    localize_inputs,
    recipe,
    reserve_cost,
    validate_config,
    write_json,
)

if sys.platform == "win32":
    from aiohttp import connector
    from aiohttp.resolver import ThreadedResolver

    # Modal's HTTP uploads use aiohttp's default resolver. On Windows, aiodns can
    # fail to reach DNS servers even when the operating system resolves the host.
    # Use system DNS in this process; cloud containers retain their Linux default.
    connector.DefaultResolver = ThreadedResolver

ROOT = (
    Path("/workspace") if Path("/workspace/src").is_dir() else Path(__file__).resolve().parents[1]
)
config_path = Path("/workspace/configs/modal_dinov3.json")
if not config_path.is_file():
    config_path = ROOT / "configs/modal_dinov3.json"
CONFIG = json.loads(config_path.read_text(encoding="utf-8"))
validate_config(CONFIG)
budget_plan(CONFIG)
REMOTE_ROOT = Path("/experiment")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.8.0",
        "torchvision==0.23.0",
        "transformers==4.57.3",
        "huggingface-hub==0.36.0",
        "numpy==2.2.6",
        "pillow==11.3.0",
        "scikit-learn==1.7.2",
        "opencv-python-headless==4.11.0.86",
        "pandas==2.2.3",
        "PyYAML==6.0.2",
        "pydantic==2.11.7",
        "requests==2.32.5",
    )
    .env(
        {
            "PYTHONPATH": "/workspace",
            "USE_TF": "0",
            "USE_FLAX": "0",
            "HF_HOME": "/experiment/hf_cache",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        }
    )
    .workdir("/workspace")
    # Keep local mounts last: subsequent build steps are rejected by Modal.
    .add_local_dir(ROOT / "src", "/workspace/src", ignore=["**/__pycache__/**", "**/*.pyc"])
    .add_local_file(ROOT / "configs/modal_dinov3.json", "/workspace/configs/modal_dinov3.json")
)
app = modal.App(APP_NAME, image=image)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
options = {
    "volumes": {str(REMOTE_ROOT): volume},
    "cpu": (2.0, 2.0),
    "memory": (16384, 16384),
    "max_containers": 1,
    "retries": 0,
    "scaledown_window": 2,
    "startup_timeout": CONFIG["startup_allowance_seconds"],
}
# Create this Secret in Modal's dashboard: HF_TOKEN; optionally KAGGLE_USERNAME/KAGGLE_KEY.
access = modal.Secret.from_name("pneumonia-dinov3-access", required_keys=["HF_TOKEN"])


@app.function(
    secrets=[access],
    cpu=(0.125, 0.125),
    memory=(512, 512),
    max_containers=1,
    retries=0,
    scaledown_window=2,
    timeout=60,
    startup_timeout=CONFIG["startup_allowance_seconds"],
)
def check_access_remote() -> dict:
    """A small billed CPU check; no GPU, model download or training reservation."""
    return {
        "model_id": CONFIG["model_id"],
        "model_revision": CONFIG["model_revision"],
        **check_model_access(CONFIG, os.environ["HF_TOKEN"]),
    }


@app.function(**options, secrets=[access], timeout=CONFIG["prepare_timeout_seconds"])
def prepare_remote(payload: dict, cohort_id: str) -> dict:
    """Download on a CPU container and retain only byte-verified development images."""
    from huggingface_hub import snapshot_download

    access_status = check_model_access(CONFIG, os.environ["HF_TOKEN"])
    if not access_status["access_granted"]:
        raise RuntimeError(
            f"Model access check failed ({access_status['status']}): {access_status['message']}"
        )
    volume.reload()
    cohort = REMOTE_ROOT / "cohorts" / cohort_id
    cohort.mkdir(parents=True, exist_ok=True)
    ready = cohort / "prepared.json"
    if ready.exists():
        prepared = json.loads(ready.read_text(encoding="utf-8"))
        if (
            prepared["model_revision"] == CONFIG["model_revision"]
            and prepared.get("bundle_sha256")
            and prepared.get("weight_files_sha256")
        ):
            return prepared
    deadline = time.monotonic() + CONFIG["prepare_timeout_seconds"] - 180
    archive = REMOTE_ROOT / "chest_xray.zip"
    if not archive.exists():
        archive_sha = download_dataset(CONFIG["dataset_url"], archive, deadline)
    else:
        import hashlib

        with archive.open("rb") as stream:
            archive_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    audit = extract_development(archive, payload, cohort)
    bundle_sha256 = create_development_bundle(archive, payload, cohort)
    # The pinned commit and source-image hashes, rather than the mutable archive URL,
    # identify the experiment. Files incompatible with the existing audit fail closed.
    weights = REMOTE_ROOT / "weights" / CONFIG["model_revision"]
    snapshot_download(
        CONFIG["model_id"],
        revision=CONFIG["model_revision"],
        token=os.environ["HF_TOKEN"],
        local_dir=weights,
        allow_patterns=["*.json", "*.safetensors"],
    )
    result = {
        "cohort_id": cohort_id,
        "model_revision": CONFIG["model_revision"],
        "archive_sha256": archive_sha,
        "counts": {s: audit["counts"][s] for s in ("train", "val")},
        "test_images_extracted": 0,
        "original_split_preserved": True,
        "bundle_sha256": bundle_sha256,
        "weight_files_sha256": {
            path.name: file_digest(path)
            for path in weights.iterdir()
            if path.suffix in {".json", ".safetensors"}
        },
    }
    write_json(ready, result)
    volume.commit()
    return result


@app.function(**options, gpu=CONFIG["gpu"], timeout=CONFIG["pilot_timeout_seconds"])
def pilot_remote(cohort_id: str, job_id: str) -> dict:
    """Measure one registered recipe on a GPU; save no partial-cohort metrics."""
    import faulthandler

    pilot_started = time.monotonic()
    volume.reload()
    output = REMOTE_ROOT / "runs" / job_id
    output.mkdir(parents=True, exist_ok=True)

    def progress(event: dict) -> None:
        event = {**event, "elapsed_seconds": time.monotonic() - pilot_started}
        print(json.dumps(event), flush=True)
        write_json(output / "pilot_progress.json", {**event, "timestamp_unix": time.time()})
        volume.commit()

    # A blocked import, file read or worker now leaves a stack trace in Modal logs.
    faulthandler.dump_traceback_later(120, repeat=True)
    try:
        progress({"stage": "stage_inputs_on_container_disk"})
        manifests, weights = localize_inputs(
            REMOTE_ROOT / "cohorts" / cohort_id,
            REMOTE_ROOT / "weights" / CONFIG["model_revision"],
            Path("/tmp/pneumonia-dinov3") / cohort_id,  # noqa: S108
        )
        progress({"stage": "import_training"})
        from src.training.train_dinov3 import run_pilot

        staging_seconds = time.monotonic() - pilot_started
        result = run_pilot(
            CONFIG,
            manifests,
            weights,
            output,
            progress=progress,
        )
        result["setup_seconds"] += staging_seconds
        result["estimated_training_seconds"] += staging_seconds
        write_json(output / "pilot.json", result)
        volume.commit()
        return result
    except Exception as error:
        progress({"stage": "failed", "error_type": type(error).__name__})
        raise
    finally:
        faulthandler.cancel_dump_traceback_later()


@app.function(**options, gpu=CONFIG["gpu"], timeout=CONFIG["train_timeout_seconds"])
def train_remote(cohort_id: str, job_id: str) -> dict:
    """Resume a matching complete-epoch checkpoint within a bounded function call."""
    started = time.monotonic()
    from src.training.train_dinov3 import run_training

    volume.reload()
    manifests, weights = localize_inputs(
        REMOTE_ROOT / "cohorts" / cohort_id,
        REMOTE_ROOT / "weights" / CONFIG["model_revision"],
        Path("/tmp/pneumonia-dinov3") / cohort_id,  # noqa: S108
    )
    return run_training(
        CONFIG,
        manifests,
        weights,
        REMOTE_ROOT / "runs" / job_id,
        CONFIG["train_timeout_seconds"] - (time.monotonic() - started) - 60,
        commit=volume.commit,
    )


@app.function(
    volumes={str(REMOTE_ROOT): volume},
    cpu=(1.0, 1.0),
    memory=(8192, 8192),
    max_containers=1,
    retries=0,
    scaledown_window=2,
    timeout=600,
    startup_timeout=180,
)
def external_prepare_remote(payload: dict, evaluation_id: str) -> dict:
    """Prepare public external data on CPU, without any model predictions."""
    from src.cloud.dinov3_external import prepare_external

    volume.reload()
    result = prepare_external(payload, REMOTE_ROOT / "external" / evaluation_id)
    volume.commit()
    return result


@app.function(
    volumes={str(REMOTE_ROOT): volume},
    cpu=(1.0, 1.0),
    memory=(8192, 8192),
    gpu="L4",
    max_containers=1,
    retries=0,
    scaledown_window=2,
    timeout=600,
    startup_timeout=180,
)
def external_test_remote(payload: dict, selection: dict, evaluation_id: str) -> dict:
    """Score a frozen external cohort once on the existing checkpoint."""
    from src.cloud.dinov3_external import evaluate_external

    volume.reload()
    result = evaluate_external(
        payload,
        selection,
        REMOTE_ROOT,
        REMOTE_ROOT / "external" / evaluation_id,
    )
    volume.commit()
    return result


@app.function(
    volumes={str(REMOTE_ROOT): volume},
    cpu=(0.125, 0.125),
    memory=(512, 512),
    max_containers=1,
    retries=0,
    scaledown_window=2,
    timeout=28800,
    startup_timeout=600,
)
def orchestrate(action: str, payload: dict | None = None) -> dict:
    """Sole ledger writer in the deployed app; reservations precede every dispatch."""
    volume.reload()
    ledger_path = REMOTE_ROOT / "budget_ledger.json"
    if action == "status":
        ledger = (
            json.loads(ledger_path.read_text(encoding="utf-8")) if ledger_path.exists() else None
        )
        reports = {}
        for filename in ("pilot.json", "pilot_progress.json", "result.json"):
            for path in (REMOTE_ROOT / "runs").glob(f"*/{filename}"):
                reports[f"{path.parent.name}/{filename}"] = json.loads(
                    path.read_text(encoding="utf-8")
                )
        return {"ledger": ledger, "reports": reports, "plan": budget_plan(CONFIG)}
    if action in {"external-prepare", "external-test"} and payload is not None:
        from src.cloud.dinov3_external import external_id, external_stage_cost, validate_selection

        selection = payload.get("selection")
        source_payload = {key: payload[key] for key in ("protocol", "references")}
        evaluation_id = external_id(source_payload)
        output = REMOTE_ROOT / "external" / evaluation_id
        report_path = output / ("prepared.json" if action == "external-prepare" else "result.json")
        if report_path.exists():
            if action == "external-test":
                locked = json.loads((output / "locked_inference.json").read_bytes())
                if locked["pre_prediction_amendment"] != selection:
                    raise ValueError("Completed external selection cannot be changed")
            return json.loads(report_path.read_bytes())
        if action == "external-test" and not selection:
            raise ValueError("Require a reviewed external selection before inference")
        if action == "external-test":
            validate_selection(
                source_payload,
                json.loads((output / "prepared.json").read_bytes()),
                selection,
            )
        # Check that the whole two-call workflow fits before starting acquisition.
        ledger = json.loads(ledger_path.read_bytes())
        needed = external_stage_cost(CONFIG, "external_coordinator")
        needed += external_stage_cost(CONFIG, "external_test")
        if action == "external-prepare":
            needed += external_stage_cost(CONFIG, "external_prepare")
            needed += external_stage_cost(CONFIG, "external_coordinator")
        if ledger["reserved_usd"] + needed > ledger["ceiling_usd"] + 1e-8:
            raise ValueError("Budget exhausted for the bounded external workflow")
        reserve_cost(ledger_path, CONFIG, "external_coordinator", evaluation_id)
        stage = "external_prepare" if action == "external-prepare" else "external_test"
        reserve_cost(ledger_path, CONFIG, stage, evaluation_id)
        volume.commit()
        if action == "external-prepare":
            result = external_prepare_remote.remote(source_payload, evaluation_id)
        else:
            result = external_test_remote.remote(source_payload, selection, evaluation_id)
        volume.reload()
        result["budget_ledger"] = json.loads(ledger_path.read_bytes())
        return result
    if action not in {"prepare", "pilot", "train", "all"} or payload is None:
        raise ValueError("Require prepare/pilot/train/all and audited development payload")
    cohort_id = digest(json.dumps(payload, sort_keys=True).encode())[:16]
    job_id = digest(
        json.dumps({"cohort": cohort_id, "recipe": recipe(CONFIG)}, sort_keys=True).encode()
    )[:16]
    cohort = REMOTE_ROOT / "cohorts" / cohort_id
    output = REMOTE_ROOT / "runs" / job_id
    reserve_cost(ledger_path, CONFIG, "coordinator", job_id)
    volume.commit()

    def funded(stage: str) -> None:
        reserve_cost(ledger_path, CONFIG, stage, job_id)
        # A remote crash still consumes its entire reservation; there are no automatic retries.
        volume.commit()

    results = {"job_id": job_id, "cohort_id": cohort_id}
    prepared_path = cohort / "prepared.json"
    prepared = (
        json.loads(prepared_path.read_text(encoding="utf-8")) if prepared_path.exists() else None
    )
    if (
        not prepared
        or prepared["model_revision"] != CONFIG["model_revision"]
        or not prepared.get("bundle_sha256")
        or not prepared.get("weight_files_sha256")
    ):
        funded("prepare")
        results["prepare"] = prepare_remote.remote(payload, cohort_id)
        volume.reload()
    if action == "prepare":
        return results
    pilot_file = output / "pilot.json"
    pilot = json.loads(pilot_file.read_text(encoding="utf-8")) if pilot_file.exists() else None
    if not pilot or pilot.get("timing_schema") != PILOT_TIMING_SCHEMA:
        funded("pilot")
        results["pilot"] = pilot_remote.remote(cohort_id, job_id)
        volume.reload()
    pilot = json.loads(pilot_file.read_text(encoding="utf-8"))
    results["pilot"] = pilot
    if action == "pilot":
        return results
    result_file = output / "result.json"
    previous = json.loads(result_file.read_text(encoding="utf-8")) if result_file.exists() else None
    if previous and previous["status"] in {"complete", "early_stopped"}:
        results["train"] = previous
        return results
    # Avoid spending hours on a recipe that the pilot already forecasts cannot finish.
    completed = previous["completed_epochs"] if previous else 0
    predicted = pilot.get("setup_seconds", 0) + pilot["conservative_epoch_seconds"] * (
        CONFIG["epochs"] - completed
    )
    if predicted > CONFIG["train_timeout_seconds"] - 300:
        results["train"] = {
            "status": "pilot_exceeds_time_limit",
            "estimated_seconds": predicted,
            "message": "Reduce registered epochs or resolution before a new run",
        }
        return results
    funded("train")
    results["train"] = train_remote.remote(cohort_id, job_id)
    volume.reload()
    results["budget_ledger"] = json.loads(ledger_path.read_text(encoding="utf-8"))
    return results


@app.local_entrypoint()
def main(action: str = "plan", manifests: str = "data/processed/clean_v1") -> None:
    """Plan/check locally; execution always uses the already deployed coordinator."""
    if action == "plan":
        print(json.dumps(budget_plan(CONFIG), indent=2))
        return
    if action == "check":
        payload = build_payload(ROOT / manifests)
        print(
            json.dumps(
                {
                    "payload_bytes": len(json.dumps(payload).encode()),
                    "contains": ["train CSV", "val CSV", "existing split audit"],
                    "plan": budget_plan(CONFIG),
                },
                indent=2,
            )
        )
        return
    if action == "access":
        checker = modal.Function.from_name(APP_NAME, "check_access_remote")
        print(json.dumps(checker.remote(), indent=2))
        return
    if action in {"external-prepare", "external-test"}:
        from src.cloud.dinov3_external import build_external_payload, external_id

        payload = build_external_payload(ROOT)
        output = ROOT / "reports/metrics/modal_dinov3_external_v1"
        if action == "external-test":
            selection = json.loads((output / "selection.json").read_bytes())
            if selection["external_id"] != external_id(payload):
                raise ValueError("Local external selection/source mismatch")
            payload["selection"] = selection
        coordinator = modal.Function.from_name(APP_NAME, "orchestrate")
        call = coordinator.spawn(action, payload)
        print(f"Remote external coordinator call: {call.object_id}", flush=True)
        result = call.get()
        write_json(
            output / ("prepared.json" if action == "external-prepare" else "result.json"), result
        )
        print(
            json.dumps(
                {
                    k: v
                    for k, v in result.items()
                    if k not in {"records", "archive_members", "budget_ledger", "excluded"}
                },
                indent=2,
            )
        )
        if result.get("excluded"):
            print(
                json.dumps(
                    {
                        "excluded_counts": {
                            dataset: sum(row["dataset"] == dataset for row in result["excluded"])
                            for dataset in sorted({row["dataset"] for row in result["excluded"]})
                        }
                    }
                )
            )
        return
    if action not in {"prepare", "pilot", "train", "all", "status"}:
        raise ValueError("Choose plan, check, access, prepare, pilot, train, all or status")
    payload = None if action == "status" else build_payload(ROOT / manifests)
    coordinator = modal.Function.from_name(APP_NAME, "orchestrate")
    call = coordinator.spawn(action, payload)
    print(f"Remote coordinator call: {call.object_id}", flush=True)
    print(json.dumps(call.get(), indent=2))
