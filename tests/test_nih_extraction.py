"""Verify source integrity and safe NIH extraction before model training."""

import gzip
import io
import json
import tarfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from src.data.download_nih import file_sha256
from src.data.extract_nih import extract_archive, image_name, inspect_image


def sample() -> tuple[bytes, dict]:
    source = io.BytesIO()
    Image.fromarray(np.tile(np.arange(256, dtype=np.uint8), (1024, 4))).save(source, format="PNG")
    row = {
        "Image Index": "00000001_000.png",
        "Patient ID": "1",
        "Patient Age": "50",
        "Finding Labels": "Effusion|Pneumonia",
        "View Position": "PA",
        "OriginalImage[Width": "2992",
        "Height]": "2991",
    }
    return source.getvalue(), row


@pytest.mark.parametrize("name", ["../images/00000001_000.png", "/images/00000001_000.png"])
def test_archive_paths_cannot_escape(name: str) -> None:
    member = tarfile.TarInfo(name)
    member.size = 50
    with pytest.raises(ValueError, match="archive member"):
        image_name(member)


def test_archive_link_is_rejected() -> None:
    member = tarfile.TarInfo("images/00000001_000.png")
    member.type = tarfile.SYMTYPE
    member.size = 50
    with pytest.raises(ValueError, match="archive member"):
        image_name(member)


def test_adult_image_preserves_source_bytes_and_pathology_label(tmp_path: Path) -> None:
    payload, row = sample()
    record, thumbnail = inspect_image(payload, row, tmp_path)
    assert Path(record["filepath"]).read_bytes() == payload
    assert record["label"] == "REPORT_PNEUMONIA"
    assert record["patient_proxy"] == "NIH:00000001"
    assert record["width"] == 1024
    assert record["metadata_original_width"] == 2992
    assert thumbnail.shape == (64, 64)


def test_minor_is_audited_without_becoming_an_adult_training_file(tmp_path: Path) -> None:
    payload, row = sample()
    row["Patient Age"] = "17"
    record, _ = inspect_image(payload, row, tmp_path)
    assert record["eligible_adult"] is False
    assert record["filepath"] is None
    assert not list(tmp_path.iterdir())


def test_existing_image_is_never_overwritten(tmp_path: Path) -> None:
    payload, row = sample()
    existing = tmp_path / row["Image Index"]
    existing.write_bytes(b"preserve existing work")
    with pytest.raises(ValueError, match="preserving"):
        inspect_image(payload, row, tmp_path)
    assert existing.read_bytes() == b"preserve existing work"


def archive_fixture(tmp_path: Path, corrupt_crc: bool = False) -> tuple[Path, dict, dict]:
    payload, row = sample()
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w") as tar:
        member = tarfile.TarInfo("images/" + row["Image Index"])
        member.size = len(payload)
        tar.addfile(member, io.BytesIO(payload))
    encoded = bytearray(gzip.compress(raw.getvalue()))
    if corrupt_crc:
        encoded[-8] ^= 1
    archive = tmp_path / "images_001.tar.gz"
    archive.write_bytes(encoded)
    expected = {
        "name": archive.name,
        "expected_bytes": len(encoded),
        "url": "https://nihcc.box.com/shared/static/example.gz",
    }
    archive.with_suffix(".gz.provenance.json").write_text(
        json.dumps({"sha256": file_sha256(archive), "source_url": expected["url"]})
    )
    return archive, expected, {row["Image Index"]: row}


def test_valid_archive_has_verified_pixels_and_artifact_hashes(tmp_path: Path) -> None:
    archive, expected, rows = archive_fixture(tmp_path)
    result = extract_archive(archive, expected, rows, tmp_path / "images", tmp_path / "audit")
    assert result["gzip_crc_verified"] is True
    assert result["adult_images"] == 1
    audit = tmp_path / "audit" / "images_001"
    for name, digest in result["artifact_sha256"].items():
        assert file_sha256(audit / name) == digest


def test_corrupt_gzip_footer_never_produces_completed_audit(tmp_path: Path) -> None:
    archive, expected, rows = archive_fixture(tmp_path, corrupt_crc=True)
    with pytest.raises(gzip.BadGzipFile):
        extract_archive(archive, expected, rows, tmp_path / "images", tmp_path / "audit")
    assert not (tmp_path / "audit" / "images_001" / "image_audit.json").exists()
