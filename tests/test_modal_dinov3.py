"""Cloud cost guards, portable split preservation and offline DINOv3 training checks."""

from __future__ import annotations

import asyncio
import csv
import importlib.util
import json
import socket
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest
import torch
from PIL import Image
from src.cloud.dinov3_job import (
    budget_plan,
    build_payload,
    create_development_bundle,
    development_rows,
    digest,
    extract_development,
    file_digest,
    image_key,
    localize_inputs,
    modal_resource_name,
    reserve_cost,
    stage_cost,
    validate_config,
    write_json,
)


@pytest.fixture
def config() -> dict:
    return json.loads(Path("configs/modal_dinov3.json").read_text(encoding="utf-8"))


@pytest.fixture
def cohort(tmp_path: Path) -> tuple[Path, dict, Path]:
    """A miniature original audit and ZIP, including an excluded test member."""
    manifests = tmp_path / "original"
    manifests.mkdir()
    archive = tmp_path / "dataset.zip"
    hashes, counts = {}, {}
    with zipfile.ZipFile(archive, "w") as zipped:
        for split_index, split in enumerate(("train", "val")):
            rows = []
            for index in range(4):
                label_id = index % 2
                label = "PNEUMONIA" if label_id else "NORMAL"
                image = tmp_path / "raw" / split / label / f"IM-{split_index}{index}-0001.png"
                image.parent.mkdir(parents=True, exist_ok=True)
                pixels = np.full((32, 32), 30 + index * 40 + split_index, dtype=np.uint8)
                Image.fromarray(pixels).save(image)
                rows.append(
                    {
                        "filepath": str(image),
                        "split": split,
                        "label": label,
                        "label_id": str(label_id),
                        "patient_proxy": f"{split}-{index}",
                        "group_id": f"{split}-{index}",
                        "sha256": digest(image.read_bytes()),
                        "pixel_sha256": digest(str(pixels.shape).encode() + pixels.tobytes()),
                    }
                )
                zipped.write(image, f"chest_xray/chest_xray/{split}/{label}/{image.name}")
            path = manifests / f"{split}_manifest.csv"
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            hashes[split] = digest(path.read_bytes())
            counts[split] = {"images": 4}
        zipped.writestr("chest_xray/test/NORMAL/IM-test.png", b"do not extract")
        zipped.writestr("../../outside.txt", b"do not extract")
    audit = {
        "manifest_sha256": hashes,
        "counts": counts,
        "cross_split_near_duplicates": 0,
        "overlaps": {
            key: {"patient_proxy": 0, "connected_components": 0}
            for key in ("train/val", "train/test", "val/test")
        },
    }
    write_json(manifests / "split_audit.json", audit)
    return manifests, build_payload(manifests), archive


def test_initial_plan_includes_gpu_cpu_ram_and_reserve(config: dict) -> None:
    plan = budget_plan(config)
    assert plan["total_reserved_usd"] == pytest.approx(14.5141)
    assert plan["total_reserved_usd"] + plan["unallocated_reserve_usd"] < 28
    assert stage_cost(config, "train") > (
        config["train_timeout_seconds"] * config["gpu_usd_per_second"]
    )


def test_modal_resource_names_support_per_workspace_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MODAL_DINOV3_APP_NAME", raising=False)
    default = modal_resource_name("MODAL_DINOV3_APP_NAME", "pneumonia-dinov3")
    assert default == "pneumonia-dinov3"
    monkeypatch.setenv("MODAL_DINOV3_APP_NAME", "my-dinov3-run")
    overridden = modal_resource_name("MODAL_DINOV3_APP_NAME", "pneumonia-dinov3")
    assert overridden == "my-dinov3-run"


@pytest.mark.parametrize(
    "change",
    [
        {"budget_usd": 29},
        {"reserve_usd": 0},
        {"gpu_usd_per_second": float("nan")},
        {"train_timeout_seconds": 90000},
        {"gpu": "H100"},
    ],
)
def test_unsafe_cost_configuration_is_rejected(config: dict, change: dict) -> None:
    with pytest.raises(ValueError):
        validate_config({**config, **change})


def test_failures_and_new_job_names_cannot_reset_cumulative_budget(
    tmp_path: Path, config: dict
) -> None:
    ledger = tmp_path / "budget.json"
    reserve_cost(ledger, config, "train", "first-attempt")
    previous = ledger.read_bytes()
    with pytest.raises(ValueError, match="Budget exhausted"):
        reserve_cost(ledger, config, "train", "different-job")
    assert ledger.read_bytes() == previous
    with pytest.raises(ValueError, match="cannot be changed"):
        reserve_cost(ledger, {**config, "budget_usd": 27}, "pilot", "another")


def test_payload_is_small_and_never_requires_test_manifest(cohort: tuple) -> None:
    manifests, payload, _ = cohort
    assert not (manifests / "test_manifest.csv").exists()
    assert set(payload["manifests"]) == {"train", "val"}
    assert len(development_rows(payload)["train"]) == 4
    assert image_key(r"C:\project\data\train\NORMAL\IM-0001.jpeg") == "train/NORMAL/IM-0001.jpeg"
    with pytest.raises(ValueError, match="test images"):
        image_key("data/test/NORMAL/example.jpeg")


def test_modified_or_overlapping_payload_is_rejected(cohort: tuple) -> None:
    _, payload, _ = cohort
    altered = json.loads(json.dumps(payload))
    altered["manifests"]["train"] += "\n"
    with pytest.raises(ValueError, match="hash"):
        development_rows(altered)
    altered = json.loads(json.dumps(payload))
    altered["audit"]["overlaps"]["train/test"]["patient_proxy"] = 1
    with pytest.raises(ValueError, match="leakage"):
        development_rows(altered)


def test_remote_extraction_preserves_split_and_bytes_without_test_images(
    tmp_path: Path,
    cohort: tuple,
) -> None:
    _, payload, archive = cohort
    target = tmp_path / "remote"
    target.mkdir()
    audit = extract_development(archive, payload, target)
    assert not (target / "images/test").exists()
    assert not (tmp_path / "outside.txt").exists()
    for split in ("train", "val"):
        with (target / f"{split}_manifest.csv").open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 4
        assert [row["group_id"] for row in rows] == [f"{split}-{i}" for i in range(4)]
        assert all(digest(Path(row["filepath"]).read_bytes()) == row["sha256"] for row in rows)
        assert audit["source_manifest_sha256"][split] == payload["audit"]["manifest_sha256"][split]
        assert audit["manifest_sha256"][split] == digest(
            (target / f"{split}_manifest.csv").read_bytes()
        )


def test_archive_mismatch_fails_before_training(tmp_path: Path, cohort: tuple) -> None:
    _, payload, archive = cohort
    changed = tmp_path / "changed.zip"
    with zipfile.ZipFile(archive) as source, zipfile.ZipFile(changed, "w") as destination:
        for member in source.infolist():
            content = source.read(member)
            if member.filename.endswith("IM-00-0001.png"):
                content = b"corrupted image"
            destination.writestr(member.filename, content)
    with pytest.raises(ValueError, match="differs from original"):
        extract_development(changed, payload, tmp_path / "rejected")


def test_mirrored_archive_extracts_each_audited_image_once(tmp_path: Path, cohort: tuple) -> None:
    _, payload, archive = cohort
    mirrored = tmp_path / "mirrored.zip"
    with zipfile.ZipFile(archive) as source, zipfile.ZipFile(mirrored, "w") as destination:
        for member in source.infolist():
            content = source.read(member)
            destination.writestr(member.filename, content)
            if member.filename.startswith("chest_xray/chest_xray/"):
                destination.writestr(member.filename.removeprefix("chest_xray/"), content)
    output = tmp_path / "extracted"
    audit = extract_development(mirrored, payload, output)
    assert audit["source_manifest_sha256"] == payload["audit"]["manifest_sha256"]
    images = list((output / "images").rglob("*.png"))
    assert len(images) == 8
    assert not (output / "images/test").exists()
    for rows in development_rows(payload).values():
        for row in rows:
            assert (
                digest((output / "images" / image_key(row["filepath"])).read_bytes())
                == row["sha256"]
            )


@pytest.mark.parametrize("corrupt_first", [False, True])
def test_conflicting_mirror_is_rejected_in_any_archive_order(
    tmp_path: Path, cohort: tuple, corrupt_first: bool
) -> None:
    _, payload, archive = cohort
    conflicting = tmp_path / "conflicting.zip"
    mirror = "chest_xray/train/NORMAL/IM-00-0001.png"
    with zipfile.ZipFile(archive) as source, zipfile.ZipFile(conflicting, "w") as destination:
        for member in source.infolist():
            duplicate = member.filename.endswith("IM-00-0001.png")
            if duplicate and corrupt_first:
                destination.writestr(mirror, b"conflicting bytes")
            destination.writestr(member.filename, source.read(member))
            if duplicate and not corrupt_first:
                destination.writestr(mirror, b"conflicting bytes")
    output = tmp_path / "rejected"
    with pytest.raises(ValueError, match="differs from original"):
        extract_development(conflicting, payload, output)
    assert not (output / "images/train/NORMAL/IM-00-0001.png").exists()


def tiny_model(monkeypatch: pytest.MonkeyPatch) -> torch.nn.Module:
    """Use actual DINOv3 layers, random tiny weights and no network/model downloads."""
    monkeypatch.setenv("USE_TF", "0")
    monkeypatch.setenv("USE_FLAX", "0")
    pytest.importorskip("transformers")
    from src.training.dinov3_model import DINOv3Classifier
    from transformers import DINOv3ViTConfig, DINOv3ViTModel

    return DINOv3Classifier(
        DINOv3ViTModel(
            DINOv3ViTConfig(
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_register_tokens=4,
                image_size=32,
            )
        )
    )


@pytest.fixture
def bundled_inputs(tmp_path: Path, cohort: tuple) -> tuple[Path, Path, dict]:
    _, payload, archive = cohort
    directory = tmp_path / "volume_cohort"
    extract_development(archive, payload, directory)
    bundle_sha = create_development_bundle(archive, payload, directory)
    weights = tmp_path / "volume_weights"
    weights.mkdir()
    (weights / "config.json").write_bytes(b"{}")
    (weights / "model.safetensors").write_bytes(b"tiny placeholder weights")
    prepared = {
        "bundle_sha256": bundle_sha,
        "weight_files_sha256": {path.name: file_digest(path) for path in weights.iterdir()},
    }
    write_json(directory / "prepared.json", prepared)
    return directory, weights, prepared


def test_bulk_staging_preserves_audited_bytes_and_rebases_only_paths(
    tmp_path: Path, cohort: tuple, bundled_inputs: tuple
) -> None:
    from src.training.train_clean import verify_manifests

    directory, weights, _ = bundled_inputs
    cache, local_weights = localize_inputs(directory, weights, tmp_path / "container_cache")
    audit = verify_manifests(cache)
    _, payload, _ = cohort
    assert audit["source_manifest_sha256"] == payload["audit"]["manifest_sha256"]
    assert len(list((cache / "images").rglob("*.png"))) == 8
    assert not (cache / "images/test").exists()
    for split in ("train", "val"):
        with (cache / f"{split}_manifest.csv").open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        assert [row["group_id"] for row in rows] == [f"{split}-{i}" for i in range(4)]
        assert all(Path(row["filepath"]).is_relative_to(cache) for row in rows)
    assert (local_weights / "model.safetensors").read_bytes() == (
        weights / "model.safetensors"
    ).read_bytes()


@pytest.mark.parametrize("artifact", ["bundle", "weights"])
def test_bulk_staging_rejects_changed_cloud_artifacts(
    tmp_path: Path, bundled_inputs: tuple, artifact: str
) -> None:
    directory, weights, _ = bundled_inputs
    source = (
        directory / "development.tar" if artifact == "bundle" else weights / "model.safetensors"
    )
    with source.open("ab") as stream:
        stream.write(b"changed after preparation")
    cache = tmp_path / "rejected"
    with pytest.raises(ValueError, match="changed after preparation"):
        localize_inputs(directory, weights, cache)
    assert not list(cache.rglob("*.partial"))


def test_tail_training_keeps_earlier_parameters_frozen(monkeypatch: pytest.MonkeyPatch) -> None:
    model = tiny_model(monkeypatch)
    model.configure_training(1)
    logits = model(torch.randn(2, 3, 32, 32))
    torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1])).backward()
    assert all(p.grad is None for p in model.backbone.layer[0].parameters())
    assert any(p.grad is not None for p in model.backbone.layer[1].parameters())
    assert model.classifier.weight.grad is not None
    assert not model.backbone.embeddings.training


def test_complete_training_and_offline_checkpoint_roundtrip(
    tmp_path: Path,
    cohort: tuple,
    config: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = tiny_model(monkeypatch)
    from src.training import train_dinov3 as module
    from src.training.dinov3_model import DINOv3Classifier

    config = {
        **config,
        "image_size": 32,
        "batch_size": 2,
        "epochs": 2,
        "warmup_epochs": 1,
        "tail_blocks": 1,
        "num_workers": 0,
    }
    _, payload, archive = cohort
    remote, output = tmp_path / "remote", tmp_path / "run"
    remote.mkdir()
    extract_development(archive, payload, remote)
    frozen = {key: value.clone() for key, value in model.backbone.layer[0].state_dict().items()}
    result = module.run_training(
        config, remote, tmp_path / "unused", output, 1000, model_factory=lambda _: model
    )
    assert result["completed_epochs"] == 2
    assert result["status"] == "complete"
    assert result["test_access"] == "none"
    assert not (remote / "test_manifest.csv").exists()
    for key, value in model.backbone.layer[0].state_dict().items():
        torch.testing.assert_close(value.cpu(), frozen[key], rtol=0, atol=0)
    saved = torch.load(output / "best_model.pt", weights_only=False)
    restored = DINOv3Classifier.from_config(saved["backbone_config"])
    restored.load_state_dict(saved["model_state_dict"], strict=True)
    restored.eval()
    with torch.inference_mode():
        assert torch.isfinite(restored(torch.zeros(1, 3, 32, 32))).all()
    last = torch.load(output / "last.pt", weights_only=False)
    assert last["next_epoch"] == 2
    assert last["history"][-1]["epoch_seconds"] > 0
    assert last["optimizer_state_dict"]["state"]


def test_time_limited_training_resumes_and_rejects_changed_recipe(
    tmp_path: Path,
    cohort: tuple,
    config: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = tiny_model(monkeypatch)
    from src.training import train_dinov3 as module

    config = {
        **config,
        "image_size": 32,
        "batch_size": 2,
        "epochs": 2,
        "warmup_epochs": 1,
        "tail_blocks": 1,
        "num_workers": 0,
    }
    _, payload, archive = cohort
    remote, output = tmp_path / "remote", tmp_path / "run"
    remote.mkdir()
    extract_development(archive, payload, remote)
    real_clock = module.time.monotonic
    elapsed = [0.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: real_clock() + elapsed[0])

    def exhaust_time() -> None:
        elapsed[0] = 250.0

    first = module.run_training(
        config,
        remote,
        tmp_path / "unused",
        output,
        360,
        commit=exhaust_time,
        model_factory=lambda _: model,
    )
    assert first["status"] == "time_limit" and first["completed_epochs"] == 1
    elapsed[0] = 0.0
    second = module.run_training(
        config,
        remote,
        tmp_path / "unused",
        output,
        1000,
        model_factory=lambda _: tiny_model(monkeypatch),
    )
    assert second["completed_epochs"] == 2
    with pytest.raises(ValueError, match="Resume protocol changed"):
        module.run_training(
            {**config, "head_lr": 0.01},
            remote,
            tmp_path / "unused",
            output,
            1000,
            model_factory=lambda _: tiny_model(monkeypatch),
        )


def test_nonfinite_outputs_cannot_be_saved() -> None:
    from src.training.train_dinov3 import require_finite

    with pytest.raises(FloatingPointError, match="Nonfinite"):
        require_finite(torch.tensor([float("nan")]), "logits")


def test_spawned_image_workers_load_batches_after_torch_initialization(
    tmp_path: Path,
    cohort: tuple,
    config: dict,
) -> None:
    from src.training.train_dinov3 import loaders

    torch.set_num_threads(2)
    if torch.cuda.is_available():
        torch.zeros(1, device="cuda")
    _, payload, archive = cohort
    directory = tmp_path / "remote"
    extract_development(archive, payload, directory)
    train_loader, val_loader = loaders(
        directory, {**config, "image_size": 32, "batch_size": 2, "num_workers": 2}
    )
    for loader in (train_loader, val_loader):
        assert loader.multiprocessing_context.get_start_method() == "spawn"
        assert loader.timeout == 120
        images, labels = next(iter(loader))
        assert images.shape == (2, 3, 32, 32)
        assert labels.shape == (2,)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Pilot uses CUDA")
def test_actual_pilot_reports_stages_and_setup_without_quality_metrics(
    tmp_path: Path,
    cohort: tuple,
    config: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.training import train_dinov3 as module

    model = tiny_model(monkeypatch)
    monkeypatch.setattr(module.DINOv3Classifier, "from_pretrained", lambda _: model)
    _, payload, archive = cohort
    directory = tmp_path / "remote"
    extract_development(archive, payload, directory)
    events = []
    actual_clock = module.time.monotonic
    logging_seconds = [0.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: actual_clock() + logging_seconds[0])

    def slow_progress(event: dict) -> None:
        events.append(event)
        logging_seconds[0] += 10

    result = module.run_pilot(
        {**config, "image_size": 32, "batch_size": 1, "tail_blocks": 1, "num_workers": 0},
        directory,
        tmp_path / "unused",
        tmp_path / "run",
        progress=slow_progress,
    )
    assert events[0]["stage"] == "initialize_cuda_and_seed"
    assert events[-1]["stage"] == "pilot_complete"
    assert any(event["stage"] == "load_training_batch" for event in events)
    assert result["setup_seconds"] > 0
    assert result["timing_schema"] == 3
    assert result["train_batch_seconds"] < 10
    assert result["estimated_training_seconds"] > (
        result["conservative_epoch_seconds"] * config["epochs"]
    )
    assert result["partial_validation_metrics"] == "not computed"
    assert (tmp_path / "run/pilot.json").exists()


def test_failed_pilot_persists_progress_and_cancels_stack_watchdog(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("modal")
    from src.training import train_dinov3 as trainer

    name = "_test_modal_dinov3_pilot_progress"
    spec = importlib.util.spec_from_file_location(name, "scripts/modal_train_dinov3.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "REMOTE_ROOT", tmp_path)
    monkeypatch.setattr(module, "localize_inputs", lambda *args: (tmp_path, tmp_path))
    commit = Mock()
    monkeypatch.setattr(module, "volume", SimpleNamespace(reload=lambda: None, commit=commit))
    watchdog, cancel = Mock(), Mock()
    monkeypatch.setattr("faulthandler.dump_traceback_later", watchdog)
    monkeypatch.setattr("faulthandler.cancel_dump_traceback_later", cancel)

    def fail_pilot(*args: object, progress: object) -> None:
        progress({"stage": "load_pretrained_model"})
        raise RuntimeError("failed loading")

    monkeypatch.setattr(trainer, "run_pilot", fail_pilot)
    with pytest.raises(RuntimeError, match="failed loading"):
        module.pilot_remote.local("cohort", "job")
    report = json.loads((tmp_path / "runs/job/pilot_progress.json").read_text())
    assert report["stage"] == "failed"
    assert report["error_type"] == "RuntimeError"
    assert commit.call_count == 4
    assert not (tmp_path / "runs/job/pilot.json").exists()
    watchdog.assert_called_once_with(120, repeat=True)
    cancel.assert_called_once()


@pytest.mark.skipif(sys.platform != "win32", reason="Windows DNS workaround")
def test_windows_modal_upload_survives_unavailable_async_dns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use Modal's HTTP client with broken async DNS and working system DNS."""
    pytest.importorskip("modal")
    from aiohttp import connector
    from aiohttp.resolver import AsyncResolver
    from modal._utils.http_utils import _http_client_with_tls

    monkeypatch.setattr(connector, "DefaultResolver", AsyncResolver)
    async_dns = AsyncMock(side_effect=OSError("Could not contact DNS servers"))
    monkeypatch.setattr(AsyncResolver, "resolve", async_dns)
    name = "_test_modal_dinov3_dns"
    spec = importlib.util.spec_from_file_location(name, "scripts/modal_train_dinov3.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)

    async def resolve_upload_host() -> None:
        system_dns = AsyncMock(
            return_value=[
                (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("127.0.0.1", 443))
            ]
        )
        monkeypatch.setattr(asyncio.get_running_loop(), "getaddrinfo", system_dns)
        async with _http_client_with_tls(timeout=5) as session:
            hosts = await session.connector._resolve_host("upload.invalid", 443)
        assert [host["host"] for host in hosts] == ["127.0.0.1"]
        system_dns.assert_awaited_once()
        async_dns.assert_not_awaited()

    asyncio.run(resolve_upload_host())


def test_deployed_coordinator_reserves_before_dispatch_and_reuses_finished_run(
    tmp_path: Path,
    cohort: tuple,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise actual orchestration locally with zero SDK/network/GPU dispatches."""
    pytest.importorskip("modal")
    name = "_test_modal_dinov3_coordinator"
    spec = importlib.util.spec_from_file_location(name, "scripts/modal_train_dinov3.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "REMOTE_ROOT", tmp_path)
    monkeypatch.setattr(module, "volume", SimpleNamespace(reload=lambda: None, commit=lambda: None))
    calls = []

    def assert_funded(stage: str) -> None:
        ledger = json.loads((tmp_path / "budget_ledger.json").read_text(encoding="utf-8"))
        assert ledger["entries"][-1]["stage"] == stage
        assert ledger["reserved_usd"] <= 24
        calls.append(stage)

    def prepare(payload: dict, cohort_id: str) -> dict:
        assert_funded("prepare")
        directory = tmp_path / "cohorts" / cohort_id
        directory.mkdir(parents=True)
        result = {
            "model_revision": module.CONFIG["model_revision"],
            "bundle_sha256": "audited-bundle",
            "weight_files_sha256": {"model.safetensors": "pinned-weights"},
        }
        write_json(directory / "prepared.json", result)
        return result

    def pilot(cohort_id: str, job_id: str) -> dict:
        assert_funded("pilot")
        directory = tmp_path / "runs" / job_id
        directory.mkdir(parents=True)
        result = {"conservative_epoch_seconds": 60, "timing_schema": 3}
        write_json(directory / "pilot.json", result)
        return result

    def train(cohort_id: str, job_id: str) -> dict:
        assert_funded("train")
        result = {"status": "complete", "completed_epochs": 16}
        write_json(tmp_path / "runs" / job_id / "result.json", result)
        return result

    monkeypatch.setattr(module, "prepare_remote", SimpleNamespace(remote=prepare))
    monkeypatch.setattr(module, "pilot_remote", SimpleNamespace(remote=pilot))
    monkeypatch.setattr(module, "train_remote", SimpleNamespace(remote=train))
    _, payload, _ = cohort
    first = module.orchestrate.local("all", payload)
    second = module.orchestrate.local("train", payload)
    assert first["job_id"] == second["job_id"]
    assert calls == ["prepare", "pilot", "train"]
    monkeypatch.setitem(module.CONFIG, "seed", 43)
    with pytest.raises(ValueError, match="Budget exhausted"):
        module.orchestrate.local("all", payload)
    assert calls == ["prepare", "pilot", "train", "pilot"]
